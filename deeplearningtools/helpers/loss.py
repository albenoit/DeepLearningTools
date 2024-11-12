# ========================================
# FileName: loss.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A collection of helpers to compute various losses and related tools
# Warning: WARNING, many loss to be tested ! check/compare with the original papers !!!
# Note: Maybe have a look here: https://niftynet.readthedocs.io/en/dev/niftynet.layer.loss_segmentation.html
# for DeepLearningTools.
# =========================================

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

def class_weights(samples_per_class, beta=None):
  """
  Compute class weights used to balance per-class loss during optimization.

  :param samples_per_class: A numpy array containing the count of samples for each class (1D vector of length equal to the number of classes).
  :type samples_per_class: numpy.ndarray

  :param beta: The beta value used in the computation. If not provided, it is calculated as (N-1)/N, where N is the total number of samples.
  :type beta: float, optional

  :return: Class weights to be applied for each class loss.
  :rtype: numpy.ndarray
  """
  class_nb=len(samples_per_class)
  if beta == None:
    N=np.sum(samples_per_class)
    beta   = (N-1.)/N
  #print('class balancing beta', beta)
  effective_num=1. - np.power(beta, samples_per_class)
  #print('effective numbers', effective_num)
  weights= (1.-beta)/np.array(effective_num)
  weights[np.where(weights==np.inf)]=0
  #return normalized weights
  return weights/np.nansum(weights)*class_nb
  
@tf.function(reduce_retracing=True)
def get_sample_class_probabilities(one_hot_labels):
  """
  Returns the vector of class probabilities of a 3D [batch, n, labels] one-hot encoded labels tensor.

  :param one_hot_labels: A tensor of shape [batchsize, n, one-hot labels].
  :type one_hot_labels: tf.Tensor

  :return: A tensor containing class probabilities for each sample.
  :rtype: tf.Tensor
  """
  #labels_shape=one_hot_labels.get_shape().as_list()
  counts = tf.reduce_sum(one_hot_labels, axis=1)
  weights = counts/one_hot_labels.shape[1]
  return weights

@tf.function(reduce_retracing=True)
def get_per_sample_class_weights(y_true):
  """
  Returns the vector of class weights of a 3D [batch, n, labels] one-hot encoded labels tensor.

  :param y_true: A tensor of shape [batchsize, n, one-hot labels].
  :type y_true: tf.Tensor

  :return: A tensor that contains class weights for each sample.
  :rtype: tf.Tensor

  WARNING: Weights are normalized to sum to 1. When some classes are not present, the weights of other classes increase!
  """
  per_sample_class_weights = tf.math.reciprocal_no_nan(get_sample_class_probabilities(y_true))
  per_sample_class_weights/= tf.reduce_sum(per_sample_class_weights, axis=1, keepdims=True)
  return per_sample_class_weights

@tf.function(reduce_retracing=True)
def get_batch_flat_tensors(labels, logits):
  """
  Prepare logits and label batch samples in a per-sample flat shape.

  :param labels: The integer values that will be one-hot encoded internally. Shape: [batchsize, ..., 1].
  :type labels: tf.Tensor
  :param logits: The predicted logits with shape [batchsize, ..., classes].
  :type logits: tf.Tensor

  :return: y_true, the one-hot encoded labels with shape [batchsize, n, classes], where n is the dimension of the flattened samples.
            y_pred, the flattened logits with shape [batchsize, n, classes].
  :rtype: Tuple[tf.Tensor, tf.Tensor]
  """
  logits_shape = tf.shape(logits)
  flat_sample_dim=tf.reduce_prod(logits_shape[1:-1])
  y_true = tf.reshape(tf.one_hot(labels, depth=logits_shape[-1]), [logits_shape[0], flat_sample_dim, logits_shape[-1]])
  y_pred = tf.reshape(logits, [logits_shape[0], flat_sample_dim, logits_shape[-1]])
  return y_true, y_pred

@tf.function
def preds_labels_preprocess_softmax_flatten(logits, labels):
  """
  A tf.function to simplify the optimization graph:

  1. Apply softmax to the input logits.

  2. Flatten each sample of the input batch.

  :param logits: The predicted logits with shape [batchsize, ..., classes].
  :type logits: tf.Tensor
  :param labels: The integer values that will be one-hot encoded internally. Shape: [batchsize, ..., 1].
  :type labels: tf.Tensor

  :return: y_true, the flattened one-hot encoded labels with shape [batchsize, n, classes], where n is the dimension of the flattened samples.
            y_pred, the flattened softmaxed logits with shape [batchsize, n, classes].
  :rtype: Tuple[tf.Tensor, tf.Tensor]
  """
  pred_probs = tf.nn.softmax(logits, axis=-1)
  y_true, y_pred=get_batch_flat_tensors(labels, pred_probs)
  return y_true, y_pred

@tf.function
def smooth_labels(labels, factor):
  """
  Smooths the labels.

  :param labels: The input labels to be smoothed.
  :type labels: tf.Tensor
  :param factor: The smoothing factor.
  :type factor: float

  :return: The smoothed labels.
  :rtype: tf.Tensor
  """
  # smooth the labels
  labels *= (tf.constant(1., tf.float32) - factor)
  labels += (factor / labels.shape[-1])
  #tf.print('labels', labels)
  return labels

def weighted_xcrosspow_loss_softmax(logits, labels, gamma=0.3, weight_class_sample_prob=False, weight_class_global_prob=False, train_class_probs=None, name='loss'):
  """
  A cross entropy loss with a power low, same idea as for focal loss but, limits oversupression of high scores
  from https://arxiv.org/pdf/1809.00076.pdf
  added specific weightings

  :param labels: A tensor of shape [batch_size,...] with class indexes (that will be one hot encoded internally).
  :param logits: A float32 tensor of shape [batch_size,...,num_classes].
  :param gamma: The power value for the power law.
  :param weight_class_sample_prob: If True, per-sample class loss is weighted by the related class sample probability.
  :param weight_class_global_prob: If True, weight the loss with respect to the true class probabilities in the training dataset.
  :param train_class_probs: A numpy array of class weights.
  :param name: The name of the loss tensor.

  :return: A cross entropy tensor of shape [batchsize, classes]
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
  such that imbalanced data can be handled more easily
  original work: https://arxiv.org/abs/1708.02002
  --> also have a look at the proposed strategy on the last bias init.

  :param labels: A tensor of shape [batch_size,...] with class indexes (that will be one hot encoded internally).
  :param logits: A float32 tensor of shape [batch_size,...,num_classes].
  :param gamma: A scalar for focal loss gamma hyper-parameter.
  :param weight_class_sample_prob: If True, per-sample class loss is weighted by the related class sample probability.
  :param weight_class_global_prob: If True, weight the loss with respect to the true class probabilities in the training dataset.
  :param train_class_probs: A numpy array of class weights.
  :param name: The name of the loss tensor.

  :return: A scalar loss value
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
  """
  Multiclass Sørensen-Dice index measure, softmax is applied internally on the y_preds.

  :param logits: The predicted logits with shape [batchsize, ..., classes].
  :param labels: Integer values, will be one hot encoded internally [batchsize, ..., 1].
  :param weight_class_sample_prob: Set True to weight the loss with respect to sample true class probabilities.
  :param weight_class_global_prob: Set True to weight the loss with respect to train dataset true class probabilities.
  :param train_class_probs: A numpy array of class weights.
  :param name: The name of the loss tensor.

  :return: The average dice loss.
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
  """
  Multiclass Jaccard loss measure, softmax is applied internally on the y_preds.

  :param logits: The predicted logits with shape [batchsize, ..., classes].
  :param labels: Integer values, will be one hot encoded internally [batchsize, ..., 1].
  :param weight_class_sample_prob: Set True to weight the loss with respect to sample true class probabilities.
  :param weight_class_global_prob: Set True to weight the loss with respect to train dataset true class probabilities.
  :param train_class_probs: A numpy array of class weights.
  :param name: The name of the loss tensor.

  :return: The average Jaccard loss.
  """
  y_true, y_pred=preds_labels_preprocess_softmax_flatten(logits, labels)

  errors=tf.math.abs(y_true-y_pred)
  errors_sorted, perm = tf.math.top_k(errors, k=tf.shape(errors)[1], name="descending_sort_{}".format(name))

  signs = 2. * y_true - 1. # target class present : value =1 vs not present, value=-1
  print('signs, SHOULD BE FLOATS!',signs)
  errors = (1. - logits * signs) # good preds with good margins, value<0 vs bad preds AND good preds with low margins, value>0 

  return tf.reduce_mean(errors, name=name)

def multiclass_jaccard_loss_softmax(logits, labels, weight_class_sample_prob=False, weight_class_global_prob=False, train_class_probs=None, name='loss'):
  """
  Multiclass Jaccard loss measure, softmax is applied internally on the y_preds.

  :param logits: The predicted logits with shape [batchsize, ..., classes].
  :param labels: Integer values, will be one hot encoded internally [batchsize, ..., 1].
  :param weight_class_sample_prob: Set True to weight the loss with respect to sample true class probabilities.
  :param weight_class_global_prob: Set True to weight the loss with respect to train dataset true class probabilities.
  :param train_class_probs: A numpy array of class weights.
  :param name: The name of the loss tensor.

  :return: The average Jaccard measures of shape [batchsize, classes].
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
  """
  Multiclass Tversky loss measure, softmax is applied internally on the y_preds.

  Reference: https://arxiv.org/pdf/1706.05721.pdf

  :param logits: The predicted logits with shape [batchsize, ..., classes].
  :param labels: Integer values, will be one hot encoded internally [batchsize, ..., 1].
  :param alpha: The weight of the false negatives penalty, (1-alpha) will be set to weigh false positives.
  :param weight_class_sample_prob: Set True to weight the loss with respect to sample true class probabilities.
  :param weight_class_global_prob: Set True to weight the loss with respect to train dataset true class probabilities.
  :param train_class_probs: A numpy array of class weights.
  :param focal: If value 0., activate the focal loss as presented in https://arxiv.org/pdf/1810.07842.pdf,
                recommended value was 0.75.
  :param name: The name of the loss tensor.

  :return: The average Tversky loss.
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
  """
  Apply a power law on the input loss function.

  Reference: https://arxiv.org/pdf/1809.00076.pdf

  :param loss: A tensor of (preliminary weighted) metrics in the range [0,1].
  :param gamma: The exponent value for the power law.
  :param name: The name of the loss tensor.

  :return: The average of (-log(loss))^gamma.
  """
  return tf.reduce_mean(tf.math.pow(-1.*tf.math.log(loss), gamma), name=name)

#-----------------------------------------
# A loss gradient lipshitz regularizer
#-----------------------------------------

def get_IOgradient_norm_lipschitzPenalty(inputs, outputs, target_lipschitz):
  """
  Computes a loss gradient Lipschitz regularizer.

  This function calculates the norm of the gradients of the outputs with respect to the inputs and penalizes deviations from the target Lipschitz constant.

  :param inputs: The input tensor.
  :param outputs: The output tensor.
  :param target_lipschitz: The target Lipschitz constant.

  :return: The mean squared difference between the slopes (norms of the gradients) and the target Lipschitz constant.
  """
  gradients = tf.gradients(outputs, [inputs])[0]
  print('Gradient=', gradients)
  slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
  return tf.reduce_mean((slopes-target_lipschitz)**2)

#---------------------------------------------
#
#---------------------------------------------

def tensor_gram_matrix(tensor):
  """
  Returns the Gram matrix of a given tensor matrix.

  Note that the input tensor is reshaped to a 2D matrix, preserving the last dimension.

  :param tensor: The input tensor.

  :return: The Gram matrix of the tensor.
  """
  inp_shape = tensor.get_shape().as_list()
  row_dims = np.prod(inp_shape[:-1])
  col_dims = inp_shape[-1]
  w = tf.reshape(tensor, (row_dims,col_dims))
  gm = tf.linalg.matmul(a=w, b=w, transpose_a=True)
  return gm

@tf.keras.utils.register_keras_serializable()
class Regularizer_soft_orthogonality(tf.Module):
  """
  Soft orthogonalization regularizer.

  This regularizer encourages the weight matrix of a layer to have a Gram matrix close to the identity matrix, promoting orthogonality among the weights.

  :param l: Regularization factor.
  """
  def __init__(self, l=0.0001):
    """
    Initialize the Soft Orthogonalization Regularizer.

    :param l: Regularization factor.
    """
    self.l=l

  def __call__(self,x):
    """
    Apply the soft orthogonal regularization to the weights of a given layer.

    Require  the Gram matrix of the weight matrix to be close to identity.

    :param x: The weights tensor of the layer.

    :return: The weight penalty.
    """
    weights_gram_matrix=tensor_gram_matrix(x)
    I = tf.linalg.eye(weights_gram_matrix.get_shape().as_list()[0])
    gram_minus_ident = weights_gram_matrix-I
    loss= tf.reduce_sum(tf.math.square(gram_minus_ident))
    return self.l*loss

  def get_config(self):
    """
    Get the configuration of the regularizer.
    """
    return{'l':self.l}

@tf.keras.utils.register_keras_serializable()
class Regularizer_Spectral_Restricted_Isometry(tf.keras.regularizers.Regularizer):
  def __init__(self, l=0.0001, nb_filters=0):
    """
    Spectral Restricted Isometry regularizer for weights.
    
    Orthogonal regularization for weights presented here: https://arxiv.org/abs/1810.09102

      - Generally more efficient than weights_regularizer_soft_orthogonality

      - WARNING, works best at the beginning if the training but too restrictive when fine tuning and should be replaced by classical l2 weights penalty

    :param l: Regularization factor.
    :param nb_filters: The number of output features.

    SRIPv2 variant here (TO BE TESTED): https://github.com/VITA-Group/Orthogonality-in-CNNs/blob/master/SVHN/train.py
    REMINDER : Other ideas with spectral norm : https://github.com/taki0112/Spectral_Normalization-Tensorflow
    """
    super(Regularizer_Spectral_Restricted_Isometry, self).__init__()
    self.l=l
    self.nb_filters = tf.keras.backend.constant(nb_filters, tf.int32, name='nb_filters')
    w_init = tf.random_normal_initializer()
    self.v = tf.Variable(initial_value=w_init(shape=(self.nb_filters,1),dtype=tf.float32), trainable=False, name='srip_v')

  def __call__(self,x):
    """
    Args: x, the weights tensor of a given layer
    Returns the weight penalty
    """
    weights_gram_matrix=tensor_gram_matrix(x)

    Ident = tf.linalg.eye(weights_gram_matrix.get_shape().as_list()[0])
    Norm  = weights_gram_matrix - Ident

    """v1 = tf.math.multiply(Norm, self.v)
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
    """
    u=tf.math.l2_normalize(tf.random.normal(shape=(Norm.get_shape().as_list()[0],1), mean=0.0, stddev=1.0))
    v=tf.math.l2_normalize(tf.linalg.matmul(a=Norm, b=u, transpose_a=True))
    matmul_norm_v=tf.linalg.matmul(a=Norm, b=v, transpose_a=False)
    u=tf.math.l2_normalize(matmul_norm_v)
    sigma=tf.math.multiply(u, matmul_norm_v)
    loss=tf.math.square(tf.linalg.norm(sigma, ord='euclidean'))

    return self.l*loss

  def get_config(self):
    config=super(Regularizer_Spectral_Restricted_Isometry, self).get_config()
    config.update({'l': float(self.l), 'nb_filters':int(self.nb_filters)})
    return config

@tf.keras.utils.register_keras_serializable()
class Regularizer_None(tf.keras.regularizers.Regularizer):
  """
  Custom regularizer that applies no regularization.
  """
  def __call__(self, x):
    return K.constant(0., dtype=tf.float32)
  def get_config(self):
    return super(Regularizer_None, self).get_config()

@tf.keras.utils.register_keras_serializable()
class Regularizer_L1L2Ortho(tf.keras.regularizers.Regularizer):
  """
  A regularizer that combine multiple ones (testing)
  """

  def __init__(self, l1=0.0, l2=0.0, ortho=0.0, ortho_type='soft', nb_filters=0):
    """
    Custom regularizer for L1/L2 weight regularization and soft orthogonality regularization.

    This regularizer combines L1/L2 weight regularization and soft orthogonality regularization.
    The regularization factors and orthogonality type can be specified during initialization.

    :param l1: Regularization factor for L1 penalty.
    :param l2: Regularization factor for L2 penalty.
    :param ortho: Regularization factor for orthogonality penalty.
    :param ortho_type: Type of orthogonalization penalty ('soft' or 'srip').
    :param nb_filters: Number of output features (required for SRIP orthogonalization).

    :ivar l1: Regularization factor for L1 penalty.
    :ivar l2: Regularization factor for L2 penalty.
    :ivar ortho: Regularization factor for orthogonality penalty.
    :ivar nb_filters: Number of output features (required for SRIP orthogonalization).
    :ivar ortho_type: Type of orthogonalization penalty ('soft' or 'srip').
    :ivar L1L2_reg: Instance of L1L2 regularizer for L1/L2 weight regularization.
    :ivar Ortho_reg: Instance of orthogonality regularizer for soft orthogonality regularization.
    """
    super(Regularizer_L1L2Ortho, self).__init__()
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
      if self.ortho_type=='soft':
        self.Ortho_reg = Regularizer_soft_orthogonality(l=ortho)
      elif self.ortho_type=='srip':
        self.Ortho_reg = Regularizer_Spectral_Restricted_Isometry(l=ortho, nb_filters=self.nb_filters)
      else:
        raise ValueError('weights_L1L2_soft_ortho_regularizer : unexpected provided ortho_type, expeting \'soft\' or \'srip\', received '+self.ortho_type)
    else:
      self.Ortho_reg = Regularizer_None()

  def __call__(self, x):
    return self.L1L2_reg(x)+self.Ortho_reg(x)
    
  def get_config(self):
      config=super(Regularizer_L1L2Ortho, self).get_config()
      config.update({'l1': float(self.l1), 'l2': float(self.l2), 'nb_filters':int(self.nb_filters), 'ortho':float(self.ortho), 'ortho_type':str(self.ortho_type)})
      return config


def multi_loss(lossesList):
  """
  Combine multiple losses into a single loss.

  Reference: refactored from the original work of Y. Gal https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb

  :param lossesList: A list of dictionaries with keys ('loss_value', 'name') representing the losses to combine.
  :param logvars: A list of associated prediction logvars

  :return: The combined loss as a single scalar value.
  """
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
  """
  Compute the reconstruction L1 loss.

  This function computes the reconstruction L1 loss (mean absolute error).

  :param inputs: The input tensor.
  :param reconstruction: The reconstructed tensor.

  :return: The computed L1 loss.
  """
  with tf.name_scope('reconstruction_loss_l1'):
    inputs_flat = tf.layers.flatten(inputs)
    reconstruction_flat = tf.layers.flatten(reconstruction)
    # Reconstruction loss
    l1_loss=tf.reduce_mean(tf.abs(inputs_flat-reconstruction_flat))
    tf.summary.scalar('L1loss', l1_loss)
    #l1_loss=tf.Print(l1_loss, [l1_loss], message='l1_loss')
    return l1_loss

def reconstruction_loss_MSE(inputs, reconstruction):
  """
  Compute the mean squared error (MSE) loss for reconstruction.

  This function computes the MSE loss for the reconstruction of inputs and reconstruction.

  :param inputs: The input tensor.
  :param reconstruction: The reconstructed tensor.

  :return: The computed MSE loss.
  """
  with tf.name_scope('reconstruction_loss_MSE'):
    # Reconstruction loss
    mse_loss=tf.keras.losses.MSE(
                                  reconstruction,
                                  inputs,
                                )
    tf.summary.scalar('MSE_loss', mse_loss)
    return mse_loss

def reconstruction_loss_BCE(inputs, reconstruction, pos_weight=1.):
  """
  Compute the binary cross-entropy (BCE) loss for reconstruction.

  This function computes the BCE loss for the reconstruction of inputs and reconstruction.

  :param inputs: The input tensor.
  :param reconstruction: The reconstructed tensor.
  :param pos_weight: The weight to assign to the positive class in the BCE loss. Default is 1.

  :return: The computed BCE loss.
  """
  with tf.name_scope('reconstruction_loss_BCE'):
    inputs_flat = tf.layers.flatten(inputs)
    reconstruction_flat = tf.layers.flatten(reconstruction)
    """xcross_loss=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( targets=inputs_flat,
                                              logits=reconstruction_flat,
                                              pos_weight=pos_weight))

    xcross_loss=tf.reduce_mean(xcross_loss)
    """
    xcross_loss=tf.reduce_mean(K.binary_crossentropy(inputs_flat,reconstruction_flat))

    tf.summary.scalar('reconstruction_loss_BCE', xcross_loss)
    #xcross_loss=tf.Print(xcross_loss, [xcross_loss], message='xcross_loss')
    return xcross_loss

def reconstruction_loss_BCE_soft(inputs, reconstruction, w=0.8):
  """
  Compute the binary cross-entropy (BCE) loss with soft labels for reconstruction.

  This function computes the BCE loss with soft labels for the reconstruction of inputs and reconstruction.

  :param inputs: The input tensor.
  :param reconstruction: The reconstructed tensor.
  :param w: The weight for balancing the loss between white and non-white pixels. Default is 0.8.

  :return: The computed BCE loss with soft labels.
  """
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
  """
  Compute the KL divergence loss between two distributions.

  This function computes the KL divergence loss between two distributions, an identifier 'id' is considered for visibility on Tensorboard 
  with tf.name_scope('kl_Divergence_loss').

  :param z_mean: The mean of the distribution.
  :param logvar: The log variance of the distribution.
  :param id: An identifier for visibility on TensorBoard.

  :return: The computed KL divergence loss.
  """
  with tf.name_scope('kl_Divergence_loss'):
    # KL Divergence loss
    kl_div_loss = 1. + logvar - tf.square(z_mean) - tf.exp(logvar)
    kl_div_loss = tf.reduce_mean(-0.5 * tf.reduce_mean(kl_div_loss, 1))
    tf.summary.scalar('VAE_kl_loss_'+str(id), kl_div_loss)
    #raw_input('VAE_kl_loss:'+str(kl_div_loss))
    return kl_div_loss

def generateTheta(L,endim):
  """
  Generate L random samples from the unit (endim)-dimensional space.

  This function generates L random samples from the unit (endim)-dimensional space.

  :param L: The number of samples to generate.
  :param endim: The dimension of the samples.

  :return: Generated random samples from the unit (endim)-dimensional space.
  """
  theta=[w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(L,endim))]
  return np.asarray(theta, dtype=np.float32)

def generateZ_ring(batchsize):
  """
  Generate samples from a ring distribution in a 2-dimensional space.

  This function generates 2D samples from a ring distribution in a 2-dimensional space.

  :param batchsize: The number of samples to generate.

  :return: Generated samples from the ring distribution.
  """
  from sklearn.datasets import make_circles
  temp=make_circles(2*batchsize,noise=.01)
  return np.squeeze(temp[0][np.argwhere(temp[1]==0),:]).astype(np.float32)

def generateZ_circle(batchsize):
  """
  Generate samples from a circle distribution in a 2-dimensional space.

  This function generates 2D samples from a circle distribution in a 2-dimensional space.

  :param batchsize: The number of samples to generate.

  :return: Generated samples from the circle distribution.
  """
  r=np.random.uniform(size=(batchsize))
  theta=2*np.pi*np.random.uniform(size=(batchsize))
  x=r*np.cos(theta)
  y=r*np.sin(theta)
  z_=np.array([x,y], dtype=np.float32).T
  return z_

def slicedWasserteinLoss_single(code, target_z, sample_points, batch_size):
  """
  Calculate the Sliced Wasserstein loss for a single code.

  This function computes the Sliced Wasserstein loss for a single code based on the given target z samples and sample points.

  :param code: The code for which to calculate the loss.
  :param target_z: The target z samples.
  :param sample_points: The sample points for projection.
  :param batch_size: The batch size.

  :return: The Sliced Wasserstein loss for the single code.
  """
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
  """
  Calculate the Sliced Wasserstein Autoencoder (AE) loss.

  This function computes the Sliced Wasserstein Autoencoder (AE) loss based on the given AE codes and target z samples.

  :param code_list: A list of AE codes.
  :param target_z: The target z samples.
  :param batch_size: The batch size.
  :param L: The number of sample points to project on (default: 50).

  :return: The Sliced Wasserstein Autoencoder (AE) loss.
  """
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

def discrepancy_slice_wasserstein(p1, p2):
  """
  Calculate the slice Wasserstein discrepancy between two distributions.

  This function computes the slice Wasserstein discrepancy between two distributions `p1` and `p2` using random projections.

  Reference: https://github.com/apple/ml-cvpr2019-swd
  
  :param p1: The first distribution tensor.
  :param p2: The second distribution tensor.

  :return: The slice Wasserstein discrepancy.
  """
  def sort_rows(matrix, num_rows):
    """
    Sort the rows of a matrix in descending order.

    :param matrix: The input matrix.
    :param num_rows: The number of rows to keep.

    :return: The sorted matrix.
    """
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

class UncorrelatedFeaturesConstraint(tf.keras.constraints.Constraint):
  """
  Uncorrelated Features Constraint.

  This constraint encourages the features of a layer to be uncorrelated by penalizing the covariance matrix deviation from the identity matrix.

  :param encoding_dim: The dimension of the encoded features.
  :param weightage: The weightage of the constraint penalty.

  Usage:
  ```python
  constraint = UncorrelatedFeaturesConstraint(encoding_dim, weightage=1.0)
  ```

  """
  def __init__(self, encoding_dim, weightage=1.0):
    """
    Initialize the Uncorrelated Features Constraint.

    :param encoding_dim: The dimension of the encoded features.
    :param weightage: The weightage of the constraint penalty.
    """
    self.encoding_dim = encoding_dim
    self.weightage = weightage

  def get_covariance(self, x):
    """
    Calculate the covariance matrix of the input tensor.

    :param x: The input tensor.

    :return: The covariance matrix.
    """
    x_centered_list = []

    for i in range(self.encoding_dim):
        x_centered_list.append(x[:, i] - K.mean(x[:, i]))

    x_centered = tf.stack(x_centered_list)
    covariance = K.dot(x_centered, K.transpose(x_centered)) / tf.cast(x_centered.get_shape()[0], tf.float32)

    return covariance

  def uncorrelated_feature(self, x):
    """
    Compute the uncorrelated feature penalty.

    :param x: The input tensor.

    :return: The penalty value.
    """
    if self.encoding_dim <= 1:
        return 0.0
    else:
        output = K.sum(K.square(self.covariance - tf.math.multiply(self.covariance, K.eye(self.encoding_dim))))
        return output

  def __call__(self, x):
    """
    Apply the uncorrelated features constraint to the input tensor.
    """
    self.covariance = self.get_covariance(x)
    return self.weightage * self.uncorrelated_feature(x)
