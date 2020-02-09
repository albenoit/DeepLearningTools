''' helpers_loss, a collection of helpers to compute various losses and related tools
    @author, Alexandre Benoit, LISTIC Lab, FRANCE
'''
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

# some losses adequate with semantic segmentation
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0-dice_coef

def jaccard_loss(y_true, y_pred):
    ''' from https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    '''
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# a loss gradient lipshitz regularizer
def get_IOgradient_norm_lipschitzPenalty(inputs, outputs, target_lipschitz):
    gradients = tf.gradients(outputs, [inputs])[0]
    print('Gradient=', gradients)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    return tf.reduce_mean((slopes-target_lipschitz)**2)

################################
def tensor_gram_matrix(tensor):
  ''' returns the gram matrix of a given tensor matrix
  Note that the input tensor is reshaped to a 2D matrix, preserving the last dimension
  '''

  inp_shape = tensor.get_shape().as_list()
  row_dims = np.prod(inp_shape[:-1])
  col_dims = inp_shape[-1]
  w = tf.reshape(tensor, (row_dims,col_dims))
  gm = tf.linalg.matmul(a=w, b=w, transpose_a=True)
  return gm

def weights_regularizer_soft_orthogonality(weights):
  ''' soft orthogonal regularization for weights:
     => require the Gram matrix of the weight matrix to be close to identity
    Args: weights, the weights tensor of a given layer
    Returns the weight penalty
  '''
  weights_gram_matrix=tensor_gram_matrix(weights)
  I = tf.linalg.eye(weights_gram_matrix.get_shape().as_list()[0])
  gram_minus_ident = weights_gram_matrix-I
  loss= tf.reduce_sum(tf.math.square(gram_minus_ident))
  return loss

from tensorflow.python.keras.engine import base_layer_utils
class weights_L1L2_soft_ortho_regularizer(tf.Module):
  def __init__(self, l1=0.0, l2=0.0, ortho=0.0, ortho_type='soft', nb_filters=0):
    self.l1 = tf.keras.backend.cast_to_floatx(l1)
    self.l2 = tf.keras.backend.cast_to_floatx(l2)
    self.ortho = tf.keras.backend.cast_to_floatx(ortho)
    self.nb_filters = tf.keras.backend.constant(nb_filters, tf.int32, name='nb_filters')
    self.ortho_type=ortho_type
    w_init = tf.random_normal_initializer()
    self.v = tf.Variable(initial_value=w_init(shape=(self.nb_filters,1),dtype=tf.float32), trainable=True, name='srip_v')

  @tf.function
  def __call__(self, x):
    if not self.l1 and not self.l2 and not self.ortho:
      return K.constant(0.)
    regularization = 0.
    if self.l1:
      regularization += self.l1 * tf.math.reduce_sum(tf.math.abs(x))
    if self.l2:
      regularization += self.l2 * tf.math.reduce_sum(tf.math.square(x))
    if self.ortho:
      if self.ortho_type is 'soft':
        regularization += self.ortho * self.weights_regularizer_soft_orthogonality(x)
      elif self.ortho_type is 'srip':
        regularization += self.ortho * self.weights_regularizer_Spectral_Restricted_Isometry(x)
      else:
        raise ValueError('weights_L1L2_soft_ortho_regularizer : unexpected provided ortho_type, expeting \'soft\' or \'srip\', received '+self.ortho_type)
    return regularization

  def get_config(self):
      return {'l1': float(self.l1), 'l2': float(self.l2), 'nb_filters':int(self.nb_filters), 'ortho':float(self.ortho), 'ortho_type':str(self.ortho_type)}

  def weights_regularizer_soft_orthogonality(self, x):
    ''' soft orthogonal regularization for weights:
       => require the Gram matrix of the weight matrix to be close to identity
      Args: x, the weights tensor of a given layer
      Returns the weight penalty
    '''
    weights_gram_matrix=tensor_gram_matrix(x)
    I = tf.linalg.eye(weights_gram_matrix.get_shape().as_list()[0])
    gram_minus_ident = weights_gram_matrix-I
    loss= tf.reduce_sum(tf.math.square(gram_minus_ident))
    return loss

  def weights_regularizer_Spectral_Restricted_Isometry(self, x):
    ''' orthogonal regularization for weights presented here :  https://arxiv.org/abs/1810.09102
      => generally more efficient than weights_regularizer_soft_orthogonality
      => WARNING, works best at the beginning if the training but too
      restrictive when fine tuning and should be replaced by classical l2 weights penalty
    Args: x, the weights tensor of a given layer
    Returns the weight penalty

    REMINDER : Other ideas with spectral norm : https://github.com/taki0112/Spectral_Normalization-Tensorflow
    '''
    weights_gram_matrix=tensor_gram_matrix(x)

    Ident = tf.linalg.eye(weights_gram_matrix.get_shape().as_list()[0])
    Norm  = weights_gram_matrix - Ident

    v1 = tf.math.multiply(Norm, self.v)
    norm1 = tf.reduce_sum(tf.math.square(v1))**0.5

    v2 = tf.math.divide(v1,norm1)

    v3 = tf.math.multiply(Norm,v2)
    loss= tf.reduce_sum(tf.math.square(v3))**0.5
    return loss

def focal_loss_softmax(logits, labels,  gamma=2, reduceSum_not_reduceAverage=False, name='loss'):
    """
    Focal loss, a cross entropy like loss that favors hard examples
    ... such that imbalanced data can be handled more easily
    original work : https://arxiv.org/abs/1708.02002
    --> also have a look at the proposed strategy on the last bias init

    Args:
      labels: A tensor of shape [batch_size,...].
      logits: A float32 tensor of shape [batch_size,...,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
      reduceSum_not_reduceAverage: increase loss value by summing all individual losses instead of averaging them by default
    Returns:
      A scalar loss value
    """
    nb_classes=logits.get_shape().as_list()[-1]
    y_pred=tf.nn.softmax(logits,axis=-1) # [batch_size,num_classes]
    eps = 1e-12
    y_pred=tf.clip_by_value(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
    labels=tf.cast(tf.one_hot(labels,depth=nb_classes), tf.float32)#y_pred.shape[1])
    L=-labels*((1-y_pred)**gamma)*tf.math.log(y_pred)
    L=tf.math.reduce_mean(tf.reduce_sum(L, axis=[1,2]), name=name)
    return L


def self_balanced_focal_loss(alpha=3, gamma=2.0, name='loss'):
    """
    Original by Yang Lu:
    from https://github.com/luyanger1799/Amazing-Semantic-Segmentation/blob/master/utils/losses.py
    This is an improvement of Focal Loss, which has solved the problem
    that the factor in Focal Loss failed in semantic segmentation.
    It can adaptively adjust the weights of different classes in semantic segmentation
    without introducing extra supervised information.
    :param alpha: The factor to balance different classes in semantic segmentation.
    :param gamma: The factor to balance different samples in semantic segmentation.
    :return:
    """

    def loss(y_true, y_pred):
        # cross entropy loss
        y_pred = backend.softmax(y_pred, -1)
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred)

        # sample weights
        sample_weights = backend.max(backend.pow(1.0 - y_pred, gamma) * y_true, axis=-1)

        # class weights
        pixel_rate = backend.sum(y_true, axis=[1, 2], keepdims=True) / backend.sum(backend.ones_like(y_true),
                                                                                   axis=[1, 2], keepdims=True)
        class_weights = backend.max(backend.pow(backend.ones_like(y_true) * alpha, pixel_rate) * y_true, axis=-1)

        # final loss
        final_loss = class_weights * sample_weights * cross_entropy
        return backend.mean(backend.sum(final_loss, axis=[1, 2]), name=name)

    return loss

def multi_loss(lossesList):
  ''' refactored from the original work of Y. Gal https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
      Args:
        lossesList:the python list of dicts with keys ('loss_value', 'name') to combine
        logvars:   the list of associated prediction logvars
      Returns:
        the loss combination as a single scalar value
  '''
  output_logvars=[]
  for id, loss in enumerate(lossesList):
      #create a dedicated output log variance variable to regress
      logvar_name='task_uncertainty_'+str(loss['name'])
      log_var = tf.Variable(tf.constant(1.0, tf.float32), name=logvar_name)
      output_logvars.append(log_var)
      tf.summary.scalar(logvar_name, log_var)
      precision = tf.math.exp(-log_var)
      single_loss = precision * loss['loss_value'] + log_var
      print('single_loss='+str(single_loss))
      if id==0:
        loss_sum=single_loss
      else:
        loss_sum+=single_loss
  return tf.math.reduce_mean(loss_sum)

def reconstruction_loss_L1(inputs, reconstruction):
  with tf.name_scope('reconstruction_loss_l1'):
    inputs_flat = tf.layers.flatten(inputs)
    reconstruction_flat = tf.layers.flatten(reconstruction)
    # Reconstruction loss
    l1_loss=tf.reduce_mean(tf.abs(inputs_flat-reconstruction_flat))
    tf.summary.scalar('L1loss', l1_loss)
    #l1_loss=tf.Print(l1_loss, [l1_loss], message='l1_loss')
    return l1_loss

def reconstruction_loss_MSE(inputs, reconstruction):
  with tf.name_scope('reconstruction_loss_MSE'):
    inputs_flat = tf.layers.flatten(inputs)
    reconstruction_flat = tf.layers.flatten(reconstruction)
    # Reconstruction loss
    mse_loss=tf.keras.losses.MSE(
                                  reconstruction,
                                  inputs,
                                )
    tf.summary.scalar('MSE_loss', mse_loss)
    return mse_loss

def reconstruction_loss_BCE(inputs, reconstruction, pos_weight=1.):
  with tf.name_scope('reconstruction_loss_BCE'):
    inputs_flat = tf.layers.flatten(inputs)
    reconstruction_flat = tf.layers.flatten(reconstruction)
    '''xcross_loss=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( targets=inputs_flat,
                                              logits=reconstruction_flat,
                                              pos_weight=pos_weight))

    xcross_loss=tf.reduce_mean(xcross_loss)
    '''
    xcross_loss=tf.reduce_mean(K.binary_crossentropy(inputs_flat,reconstruction_flat))

    tf.summary.scalar('reconstruction_loss_BCE', xcross_loss)
    #xcross_loss=tf.Print(xcross_loss, [xcross_loss], message='xcross_loss')
    return xcross_loss

def reconstruction_loss_BCE_soft(inputs, reconstruction, w=0.8):
  with tf.name_scope('reconstruction_loss_BCE_soft'):
    inputs_flat = tf.layers.flatten(inputs)
    reconstruction_flat = tf.layers.flatten(reconstruction)

    # handmade binary cross entropy loss taking into account the lower representation of the white pixels
    xcross_loss=-tf.reduce_mean(w * inputs_flat * tf.math.log(reconstruction_flat + 1e-8),
                                         reduction_indices=[1]) - \
                         tf.reduce_mean(
                             (1 - w) * (1 - inputs_flat) * tf.math.log(1 - reconstruction_flat + 1e-8),
                             reduction_indices=[1])

    xcross_loss=tf.reduce_mean(xcross_loss)
    tf.summary.scalar('reconstruction_loss_BCE_soft', xcross_loss)

    return xcross_loss

def kl_loss(z_mean, logvar, id):
  ''' KL loss between two distributions,
  an identifier 'id' is considered for visibility on Tensorboard '''
  with tf.name_scope('kl_Divergence_loss'):
    # KL Divergence loss
    kl_div_loss = 1. + logvar - tf.square(z_mean) - tf.exp(logvar)
    kl_div_loss = tf.reduce_mean(-0.5 * tf.reduce_mean(kl_div_loss, 1))
    tf.summary.scalar('VAE_kl_loss_'+str(id), kl_div_loss)
    #raw_input('VAE_kl_loss:'+str(kl_div_loss))
    return kl_div_loss

def generateTheta(L,endim):
  # This function generates L random samples from the unit `ndim'-u
  theta=[w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(L,endim))]
  return np.asarray(theta, dtype=np.float32)

def generateZ_ring(batchsize):
  # This function generates 2D samples from a `circle' distribution in
  # a 2-dimensional space
  from sklearn.datasets import make_circles
  temp=make_circles(2*batchsize,noise=.01)
  return np.squeeze(temp[0][np.argwhere(temp[1]==0),:]).astype(np.float32)

def generateZ_circle(batchsize):
  # This function generates 2D samples from a `circle' distribution in
  # a 2-dimensional space
  r=np.random.uniform(size=(batchsize))
  theta=2*np.pi*np.random.uniform(size=(batchsize))
  x=r*np.cos(theta)
  y=r*np.sin(theta)
  z_=np.array([x,y], dtype=np.float32).T
  return z_

def slicedWasserteinLoss_single(code, target_z, sample_points, batch_size):
  # Let projae be the projection of the encoded samples
  projae=tf.matmul(code,tf.transpose(sample_points))
  # Let projz be the projection of the $q_Z$ samples
  #projz=K.dot(target_z,K.transpose(sample_points))
  projz=tf.matmul(target_z,tf.transpose(sample_points))
  #projz=tf.Print(projz, [projz, projz_tf], message='k vc tf')
  # Calculate the Sliced Wasserstein distance by sorting
  # the projections and calculating the L2 distance between
  W2=(tf.nn.top_k(tf.transpose(projae),k=batch_size).values-tf.nn.top_k(tf.transpose(projz),k=batch_size).values)**2
  #W2=tf.Print(W2, [tf.nn.top_k(tf.transpose(projae),k=batch_size).values[0]], message='sorted projae', summarize=10)
  w2weight=tf.Variable(tf.constant(10.0), trainable=False)
  tf.summary.scalar('W2_weight', w2weight)
  W2Loss= w2weight*tf.reduce_mean(W2)
  return W2Loss

def swae_loss(code_list, target_z, batch_size, L=50):
  '''Sliced Wasserstein Autoencodeur (AE) loss definition
    Args:
      inputs
      codes_list: a python list of AE codes
      reconstructed_data: the reconstructed data at the output of the AE
      L, the number of sample points to project on
  '''

  from tensorflow.keras.layers import Flatten
  from tensorflow.keras.layers import Reshape
  print('Setting up a Sliced Wasserstein loss following https://arxiv.org/abs/1804.01947')
  if len(code_list)==0:
    raise ValueError('swae_loss error : input code list is empty')
  if len(code_list[0].shape)!=2:
    raise ValueError('swae_loss error : input codes must be flat codes of size batchsize*codeDim')
  #generate two variables used for the loss at the current iteration
  #theta=tf.Variable(tf.ones(generateTheta(L,code_list[0].shape[-1]).shape), trainable=False) #Define a Keras Variable for \theta_ls
  #theta=tf.Print(theta, [theta], message='theta')
  #target_z=tf.Variable(tf.ones(target_z_samples.shape), trainable=False) #Define a Keras Variable for samples of z)
  theta=tf.numpy_function(generateTheta,[L, code_list[0].shape[-1]], tf.float32)
  #theta=tf.Print(theta, [theta, target_z], message='theta, target_z')

  loss=0
  for id, code in enumerate(tf.get_collection('codes')):
    W2Loss = slicedWasserteinLoss_single(code, target_z, theta, batch_size)
    tf.summary.scalar('w2loss_'+str(id), W2Loss)
    print('W2Loss='+str(W2Loss))
    loss+=W2Loss
  #loss=tf.Print(loss, [loss], message='SWAE')
  return loss


class UncorrelatedFeaturesConstraint (tf.keras.constraints.Constraint):
    ''' from https://towardsdatascience.com/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-ii-24b9cca69bd6
    '''
    def __init__(self, encoding_dim, weightage = 1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage

    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        covariance = K.dot(x_centered, K.transpose(x_centered)) / tf.cast(x_centered.get_shape()[0], tf.float32)

        return covariance

    # Constraint penalty
    def uncorrelated_feature(self, x):
        if(self.encoding_dim <= 1):
            return 0.0
        else:
            output = K.sum(K.square(
                self.covariance - tf.math.multiply(self.covariance, K.eye(self.encoding_dim))))
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)
