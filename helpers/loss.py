''' helpers_loss, a collection of helpers to compute various losses and related tools
    @author, Alexandre Benoit, LISTIC Lab, FRANCE

    WARNING, many loss to be tested ! check/compare with the original papers !!!

    NOTE,maybe have a look here: https://niftynet.readthedocs.io/en/dev/niftynet.layer.loss_segmentation.html
'''
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


def class_weights(samples_per_class, beta=None):
  ''' from https://arxiv.org/abs/1901.05555
      compute a class weight use to balance per class loss along optimization
      Args:
          a numpy array that contains dataset class samples count (1D vector of len=nb classes)
      Return:
          class weights (factor to be applied for each class loss)
  '''
  class_nb=len(samples_per_class)
  if beta is None:
    N=np.sum(samples_per_class)
    beta   = (N-1.)/N
  #print('class balancing beta', beta)
  effective_num=1. - np.power(beta, samples_per_class)
  #print('effective numbers', effective_num)
  weights= (1.-beta)/np.array(effective_num)
  weights[np.where(weights==np.inf)]=0
  #return normalized weights
  return weights/np.nansum(weights)*class_nb
  
@tf.function(experimental_relax_shapes=True)
def get_sample_class_probabilities(one_hot_labels):
  ''' returns the vector of class probabilities of 3D [batch, n, labels] one hot encoded labels tensor
      Args: one_hot_labels, a tensor of shape [batchsize, n , one hot labels]
      Returns tensor that contains class probabilities for each of the samples
  '''
  #labels_shape=one_hot_labels.get_shape().as_list()
  counts = tf.reduce_sum(one_hot_labels, axis=1)
  weights = counts/one_hot_labels.shape[1]
  return weights

@tf.function(experimental_relax_shapes=True)
def get_per_sample_class_weights(y_true):
  ''' returns the vector of class weights of 3D [batch, n, labels] one hot encoded labels tensor
      Args: one_hot_labels, a tensor of shape [batchsize, n , one hot labels]
      Returns tensor that contains class probabilities for each of the samples
      WARNING, weights are normalized to sum to 1, then when some classes are not present, the others weight increase !
  '''
  per_sample_class_weights = tf.math.reciprocal_no_nan(get_sample_class_probabilities(y_true))
  per_sample_class_weights/= tf.reduce_sum(per_sample_class_weights, axis=1, keepdims=True)
  return per_sample_class_weights

@tf.function(experimental_relax_shapes=True)
def get_batch_flat_tensors(labels, logits):
  '''
    Prepare logits and label batch samples in a per sample flat shape
    Args: 
        logits the predicted logits with shape [batchsize, ..., classes]
        labels (integer values, will be one hot encoded internally [batchsize, ..., 1]
    Returns :
        y_true, one hot encoded labels, shape=[batchsize, n, classes] with n=the dimension of the flatten samples 
        y_pred, flat logits, shape=[batchsize, n, classes]
  '''

  logits_shape = tf.shape(logits)
  flat_sample_dim=tf.reduce_prod(logits_shape[1:-1])
  y_true = tf.reshape(tf.one_hot(labels, depth=logits_shape[-1]), [logits_shape[0], flat_sample_dim, logits_shape[-1]])
  y_pred = tf.reshape(logits, [logits_shape[0], flat_sample_dim, logits_shape[-1]])
  return y_true, y_pred


@tf.function
def preds_labels_preprocess_softmax_flatten(logits, labels):
  '''
  a tf.function to simplify the optimisation graph:
  1. apply softmax to input logits
  2. flatten each sample of the input batch
  Args:
          logits the predicted logits with shape [batchsize, ..., classes]
          labels (integer values, will be one hot encoded internally [batchsize, ..., 1]
  Returns  sample flatten y_true, y_pred (softmaxed logits) tensors
  '''
  pred_probs = tf.nn.softmax(logits, axis=-1)
  y_true, y_pred=get_batch_flat_tensors(labels, pred_probs)
  return y_true, y_pred

@tf.function
def smooth_labels(labels, factor):
    # smooth the labels
    labels *= (tf.constant(1., tf.float32) - factor)
    labels += (factor / labels.shape[-1])
    #tf.print('labels', labels)
    return labels

def weighted_xcrosspow_loss_softmax(logits, labels, gamma=0.3, weight_class_sample_prob=False, weight_class_global_prob=False, train_class_probs=None, name='loss'):
    """
    a cross entropy loss with a power low, same idea as for focal loss but, limits oversupression of high scores
    from https://arxiv.org/pdf/1809.00076.pdf
    added specific weightings

    Args:
      labels: A tensor of shape [batch_size,...] with class indexes (that willbe one hot encoded internally).
      logits: A float32 tensor of shape [batch_size,...,num_classes].
      gamma
      weight_class_sample_prob: if True, per sample class loss by the related class sample probability
      weight_class_global_prob, set True to weight the loss with respect to train dataset true class probabilities
      train_class_probs, a numpy array of class weights
    Returns:
      A cross entropy tensor of shape [batchsize, classes]
    """
    y_true, y_pred=preds_labels_preprocess_softmax_flatten(logits, labels)
    
    eps=1e-10
    y_pred=tf.clip_by_value(y_pred, eps, 1.-eps)#avoid undefined values
    #standard cross entropy ^ gamma
    L=y_true*tf.math.pow(-tf.math.log(y_pred), gamma)

    if weight_class_sample_prob:
        class_rates = get_sample_class_probabilities(y_true)
        class_rates=tf.reshape(class_rates, [-1,1,y_pred.shape[-1]])
        weighting_factor=pow(tf.ones_like(L) * 3.0, class_rates)
        L*=weighting_factor
    if weight_class_global_prob:
      train_class_weights_factor=tf.constant(train_class_probs, dtype=tf.float32)
      train_class_weights_factor=tf.reshape(train_class_weights_factor, [-1,1,y_pred.shape[-1]])
      L*=train_class_weights_factor
      per_sample_loss=tf.reduce_sum(L, axis=-1)
      return tf.math.reduce_mean(per_sample_loss, name=name)
            
def focal_loss_softmax(logits, labels,  gamma=3., weight_class_sample_prob=False, weight_class_global_prob=False, train_class_probs=None, name='loss'):
    """
    Focal loss, a cross entropy like loss that favors hard examples
    ... such that imbalanced data can be handled more easily
    original work : https://arxiv.org/abs/1708.02002
    --> also have a look at the proposed strategy on the last bias init

    Args:
      labels: A tensor of shape [batch_size,...] with class indexes (that willbe one hot encoded internally).
      logits: A float32 tensor of shape [batch_size,...,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
      weight_class_sample_prob: if True, per sample class loss by the related class sample probability
      weight_class_global_prob, set True to weight the loss with respect to train dataset true class probabilities
      train_class_probs, a numpy array of class weights
    Returns:
      A scalar loss value
    """
    y_true, y_pred=preds_labels_preprocess_softmax_flatten(logits, labels)
    
    eps=1e-15
    y_pred=tf.clip_by_value(y_pred, eps, 1.-eps)#avoid undefined values
    L=-y_true*((1-y_pred)**gamma)*tf.math.log(y_pred)
    if weight_class_sample_prob:
        class_rates = get_sample_class_probabilities(y_true)
        class_rates=tf.reshape(class_rates, [-1,1,y_pred.shape[-1]])
        weighting_factor=pow(tf.ones_like(L) * 3.0, class_rates)
        L*=weighting_factor
    if weight_class_global_prob:
      train_class_weights_factor=tf.constant(train_class_probs, dtype=tf.float32)
      train_class_weights_factor=tf.reshape(train_class_weights_factor, [-1,1,y_pred.shape[-1]])
      L*=train_class_weights_factor
    per_sample_loss=tf.reduce_sum(L, axis=-1)
    return tf.math.reduce_mean(per_sample_loss, name=name)

def multiclass_dice_loss_softmax(logits, labels, weight_class_sample_prob=False, weight_class_global_prob=False, train_class_probs=None, name='loss'):
    """ multiclass Sørensen-Dice index measure, softmax is applied internally on the y_preds
      Args:
          logits the predicted logits with shape [batchsize, ..., classes]
          labels (integer values, will be one hot encoded internally [batchsize, ..., 1]
          weight_class_sample_prob, set True to weight the loss with respect to sample true class probabilities
          weight_class_global_prob, set True to weight the loss with respect to train dataset true class probabilities
          train_class_probs, a numpy array of class weights
      Returns the average dice loss
    """
    y_true, y_pred=preds_labels_preprocess_softmax_flatten(logits, labels)
    true_pos = tf.reduce_sum(    y_true * y_pred,     axis=1)
    false_pos = tf.reduce_sum( (1-y_true) * y_pred,     axis=1)
    false_neg = tf.reduce_sum(     y_true * (1-y_pred), axis=1)
    smooth=tf.keras.backend.epsilon()
    alpha=0.5
    dice_losses = 1.0-(true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

    if weight_class_sample_prob:
      dice_losses*=get_per_sample_class_weights(y_true)
    if weight_class_global_prob:
      train_class_weights_factor=tf.constant(train_class_probs, dtype=tf.float32)
      train_class_weights_factor=tf.reshape(train_class_weights_factor, [-1,y_pred.shape[-1]])
      dice_losses*=train_class_weights_factor
    
    return tf.reduce_mean(dice_losses, name=name)


def multiclass_lovasz_loss_softmax(logits, labels, weight_class_sample_prob=False, weight_class_global_prob=False, train_class_probs=None, name='loss'):
  """ multiclass Jaccard loss measure, softmax is applied internally on the y_preds
    Args: 
        logits the predicted logits with shape [batchsize, ..., classes]
        labels (integer values, will be one hot encoded internally [batchsize, ..., 1]
        weight_class_sample_prob, set True to weight the loss with respect to sample true class probabilities
        weight_class_global_prob, set True to weight the loss with respect to train dataset true class probabilities
        train_class_probs, a numpy array of class weights
    Returns the average jaccard loss
    """
  y_true, y_pred=preds_labels_preprocess_softmax_flatten(logits, labels)

  errors=tf.math.abs(y_true-y_pred)
  errors_sorted, perm = tf.math.top_k(errors, k=tf.shape(errors)[1], name="descending_sort_{}".format(c))

  signs = 2. * y_true - 1. # target class present : value =1 vs not present, value=-1
  print('signs, SHOULD BE FLOATS!',signs)
  errors = (1. - logits * signs) # good preds with good margins, value<0 vs bad preds AND good preds with low margins, value>0 


  return tf.reduce_mean(jaccard_losses, name=name)

def multiclass_jaccard_loss_softmax(logits, labels, weight_class_sample_prob=False, weight_class_global_prob=False, train_class_probs=None, name='loss'):
    """ multiclass Jaccard loss measure, softmax is applied internally on the y_preds
      Args: 
          logits the predicted logits with shape [batchsize, ..., classes]
          labels (integer values, will be one hot encoded internally [batchsize, ..., 1]
          weight_class_sample_prob, set True to weight the loss with respect to sample true class probabilities
          weight_class_global_prob, set True to weight the loss with respect to train dataset true class probabilities
          train_class_probs, a numpy array of class weights
      Returns the average jaccard measures of shape [batchsize, classes]
    """
    y_true, y_pred=preds_labels_preprocess_softmax_flatten(logits, labels)

    intersects = tf.reduce_sum(y_true * y_pred, axis=1)
    denominators = tf.reduce_sum(y_true + y_pred, axis=1)-intersects
    smooth=1.
    jaccard_losses = 1.- (intersects+smooth) / (denominators+smooth)
    print('jaccard.shape', jaccard_losses.shape)

    if weight_class_sample_prob:
      jaccard_losses*=get_per_sample_class_weights(y_true)
    if weight_class_global_prob:
      train_class_weights_factor=tf.constant(train_class_probs, dtype=tf.float32)
      train_class_weights_factor=tf.reshape(train_class_weights_factor, [-1,y_pred.shape[-1]])
      jaccard_losses*=train_class_weights_factor

    return tf.reduce_mean(jaccard_losses, name=name)

def multiclass_tversky_loss_softmax(logits, labels, alpha=0.7, weight_class_sample_prob=False, weight_class_global_prob=False, train_class_probs=None, focal=0., name='loss'):
    """ multiclass Tversky loss measure, softmax is applied internally on the y_preds
      https://arxiv.org/pdf/1706.05721.pdf
      
      Args: 
          logits, the predicted logits with shape [batchsize, ..., classes]
          labels, (integer values, will be one hot encoded internally [batchsize, ..., 1]
          alpha, the weight of the false negatives penalty, (1-alpha) will be set to weigh false positives
          weight_class_sample_prob, set True to weight the loss with respect to sample true class probabilities
          weight_class_global_prob, set True to weight the loss with respect to train dataset true class probabilities
          train_class_probs, a numpy array of class weights
          focal if value 0., activate the focal loss as presented in https://arxiv.org/pdf/1810.07842.pdf, recommended value was 0.75
      Returns the average jaccard loss
    """
    y_true, y_pred=preds_labels_preprocess_softmax_flatten(logits, labels)
    eps = tf.keras.backend.epsilon()
    y_pred=tf.clip_by_value(y_pred, eps, 1.-eps)#avoid undefined values

    true_pos = tf.reduce_sum(    y_true * y_pred,     axis=1)
    false_pos = tf.reduce_sum( (1-y_true) * y_pred,     axis=1)
    false_neg = tf.reduce_sum(     y_true * (1-y_pred), axis=1)
    smooth=eps
    tversky_losses = 1.- (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
    #tf.print('tversky_losses : tp', true_pos)
    #tf.print('tversky_losses : fp', false_pos)
    #tf.print('tversky_losses : fn', false_neg)
    #tf.print('tversky_losses : loss', tversky_losses)

    if focal>0.:
      tversky_losses=tf.math.pow(tversky_losses, focal)
      #tf.print('weighted tversky_losses : loss', tversky_losses)
    #print('tversky_losses.shape', tversky_losses.shape)

    if weight_class_sample_prob:
      tversky_losses*=get_per_sample_class_weights(y_true)
      #tf.print('weighted tversky_losses : class probs', get_per_sample_class_weights(y_true))
      #tf.print('weighted tversky_losses : loss', tversky_losses)
    
    if weight_class_global_prob:
      train_class_weights_factor=tf.constant(train_class_probs, dtype=tf.float32)
      train_class_weights_factor=tf.reshape(train_class_weights_factor, [-1,y_pred.shape[-1]])
      tversky_losses*=train_class_weights_factor

    #tf.print('tversky is finite', tf.math.reduce_prod(tf.cast(tf.math.is_finite(tversky_losses), dtype=tf.int8)))
    #tf.print('final_loss', tf.reduce_mean(tversky_losses, name=name))
    return tf.reduce_mean(tversky_losses, name=name)

def exponentialLogLoss(loss, gamma=0.3, name='loss'):
  ''' REMINDER from https://arxiv.org/pdf/1809.00076.pdf
    apply a power low on the input loss function
    Args: a tensor of (preliminary weighted) metrics in range [0,1]
    Returns the average of (-log(loss))^gamma
  '''
  return tf.reduce_mean(tf.math.pow(-1.*tf.math.log(loss), gamma), name=name)

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

class Regularizer_soft_orthogonality(tf.Module):
  def __init__(self, l=0.0001):
    ''' Soft orthogonalization regularizer
    Args:
      l, regularization factor
    '''
    self.l=l

  def __call__(self,x):
    ''' soft orthogonal regularization for weights:
      => require the Gram matrix of the weight matrix to be close to identity
    Args: x, the weights tensor of a given layer
    Returns the weight penalty
    '''
    weights_gram_matrix=tensor_gram_matrix(x)
    I = tf.linalg.eye(weights_gram_matrix.get_shape().as_list()[0])
    gram_minus_ident = weights_gram_matrix-I
    loss= tf.reduce_sum(tf.math.square(gram_minus_ident))
    return self.l*loss

  def get_config(self):
    return{'l':self.l}

class Regularizer_Spectral_Restricted_Isometry(tf.Module):
  def __init__(self, l=0.0001, nb_filters=0):
    ''' orthogonal regularization for weights presented here :  https://arxiv.org/abs/1810.09102
      => generally more efficient than weights_regularizer_soft_orthogonality
      => WARNING, works best at the beginning if the training but too
      restrictive when fine tuning and should be replaced by classical l2 weights penalty
    Args:
      l, regularization factor
      nb_filters, the number of output features


    SRIPv2 variant here (TO BE TESTED): https://github.com/VITA-Group/Orthogonality-in-CNNs/blob/master/SVHN/train.py
    REMINDER : Other ideas with spectral norm : https://github.com/taki0112/Spectral_Normalization-Tensorflow
    '''
    self.l=l
    self.nb_filters = tf.keras.backend.constant(nb_filters, tf.int32, name='nb_filters')
    w_init = tf.random_normal_initializer()
    self.v = tf.Variable(initial_value=w_init(shape=(self.nb_filters,1),dtype=tf.float32), trainable=False, name='srip_v')

  def __call__(self,x):
    '''
    Args: x, the weights tensor of a given layer
    Returns the weight penalty
    '''
    weights_gram_matrix=tensor_gram_matrix(x)

    Ident = tf.linalg.eye(weights_gram_matrix.get_shape().as_list()[0])
    Norm  = weights_gram_matrix - Ident

    '''v1 = tf.math.multiply(Norm, self.v)
    norm1 = tf.reduce_sum(tf.math.square(v1))**0.5

    v2 = tf.math.divide(v1,norm1)

    v3 = tf.math.multiply(Norm,v2)
    loss= tf.reduce_sum(tf.math.square(v3))**0.5
    #l2_reg = (torch.norm(v3,2))**2
    
    # V2 version (pytorch code):
    u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
    v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
    u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
    sigma = torch.dot(u, torch.matmul(w_tmp, v))
    loss=(torch.norm(sigma,2))**2
    '''
    u=tf.math.l2_normalize(tf.random.normal(shape=(Norm.get_shape().as_list()[0],1), mean=0.0, stddev=1.0))
    v=tf.math.l2_normalize(tf.linalg.matmul(a=Norm, b=u, transpose_a=True))
    matmul_norm_v=tf.linalg.matmul(a=Norm, b=v, transpose_a=False)
    u=tf.math.l2_normalize(matmul_norm_v)
    sigma=tf.math.multiply(u, matmul_norm_v)
    loss=tf.math.square(tf.linalg.norm(sigma, ord='euclidean'))

    return self.l*loss

  def get_config(self):
    return{'l':self.l, 'nb_filters':int(self.nb_filters)}

class Regularizer_None(tf.Module):
  def __call__(self, x):
    return K.constant(0., dtype=tf.float32)
  def get_config(self):
    return{}

class Regularizer_L1L2Ortho(tf.Module):
  ''' a regularizer that combine multiple ones (testing) '''

  def __init__(self, l1=0.0, l2=0.0, ortho=0.0, ortho_type='soft', nb_filters=0):
    '''
      l1, regularization factor for l1 penalty
      l2, regularization factor for l2 penalty
      ortho, regularization factor for orthogonality penalty
      ortho_type, the type of orthogonalization penalty
      nb_filters, the number of output features (only required for SRIP orthogonalization)
    '''
    self.l1 = tf.keras.backend.cast_to_floatx(l1)
    self.l2 = tf.keras.backend.cast_to_floatx(l2)
    self.ortho = tf.keras.backend.cast_to_floatx(ortho)
    self.nb_filters = int(nb_filters)
    self.ortho_type=ortho_type
    
    #prepare regularization operators (and no ops)
    if self.l1>0 or self.l2>0:
      self.L1L2_reg = tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2)
    else : #no op like
      self.L1L2_reg = Regularizer_None()

    if self.ortho>0:
      if self.ortho_type is 'soft':
        self.Ortho_reg = Regularizer_soft_orthogonality(l=ortho)
      elif self.ortho_type is 'srip':
        self.Ortho_reg = Regularizer_Spectral_Restricted_Isometry(l=ortho, nb_filters=self.nb_filters)
      else:
        raise ValueError('weights_L1L2_soft_ortho_regularizer : unexpected provided ortho_type, expeting \'soft\' or \'srip\', received '+self.ortho_type)
    else:
      self.Ortho_reg = Regularizer_None()

  def __call__(self, x):
    return self.L1L2_reg(x)+self.Ortho_reg(x)
    
  def get_config(self):
      return {'l1': float(self.l1), 'l2': float(self.l2), 'nb_filters':int(self.nb_filters), 'ortho':float(self.ortho), 'ortho_type':str(self.ortho_type)}


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
  for id, code in code_list:#enumerate(tf.get_collection('codes')):
    W2Loss = slicedWasserteinLoss_single(code, target_z, theta, batch_size)
    tf.summary.scalar('w2loss_'+str(id), W2Loss)
    print('W2Loss='+str(W2Loss))
    loss+=W2Loss
  #loss=tf.Print(loss, [loss], message='SWAE')
  return loss



#from https://github.com/apple/ml-cvpr2019-swd
def discrepancy_slice_wasserstein(p1, p2):

  def sort_rows(matrix, num_rows):
    matrix_T = tf.transpose(matrix, [1, 0])
    sorted_matrix_T = tf.nn.top_k(matrix_T, num_rows)[0]
    return tf.transpose(sorted_matrix_T, [1, 0])

  
  s = tf.shape(p1)
  if p1.get_shape().as_list()[1] > 1:
      # For data more than one-dimensional, perform multiple random projection to 1-D
      proj = tf.random.normal([tf.shape(p1)[1], 128])
      proj *= tf.math.rsqrt(tf.math.reduce_sum(tf.math.square(proj), 0, keepdims=True))
      p1 = tf.linalg.matmul(p1, proj)
      p2 = tf.linalg.matmul(p2, proj)
  p1 = sort_rows(p1, s[0])
  p2 = sort_rows(p2, s[0])
  wdist = tf.math.reduce_mean(tf.math.square(p1 - p2))
  return tf.math.reduce_mean(wdist)


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
