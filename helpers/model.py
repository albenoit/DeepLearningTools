import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.python.layers.core import Dense
import helpers.loss

class ReplicatedOrthogonalInitialize(tf.keras.initializers.Orthogonal):
  

  def __init__(self, scale, gain=1.0, seed=None):
      super(ReplicatedOrthogonalInitialize, self).__init__(gain, seed)
      self.scale=scale
      self.sampling_shape_factor=tf.constant([1,1,1,self.scale**2], dtype=tf.int64)

  def __call__(self, shape, dtype=tf.dtypes.float32):
    """ Overrides the parent call function
        does the same op but replicates initialization for a set of q filters
        => useful for shuffle upsampling convolution as proposed in https://arxiv.org/abs/1707.02937
    Returns a tensor object initialized as specified by the initializer.
    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported.
    Raises:
      ValueError: If the dtype is not floating point or the input shape is not
       valid.
    """
    #tf.print('ReplicatedOrthogonalInitialize: inputshape', shape)
    subsampled_shape=shape//self.sampling_shape_factor
    #tf.print('ReplicatedOrthogonalInitialize: subsampled shape', subsampled_shape)
    sampled_inits = super(ReplicatedOrthogonalInitialize, self).__call__(subsampled_shape, dtype)
    
    sampled_inits_t = tf.transpose(sampled_inits, perm=[2, 0, 1, 3])
    #tf.print('ReplicatedOrthogonalInitialize: subsampled transposed shape', sampled_inits_t.shape)
    
    sampled_inits_t_nn = tf.image.resize(sampled_inits_t,
                                         size=(shape[0] * self.scale, shape[1] * self.scale),
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #tf.print('ReplicatedOrthogonalInitialize: tiled inits transposed NN shape', sampled_inits_t_nn.shape)
    sampled_inits_t_nn = tf.nn.space_to_depth(sampled_inits_t_nn, block_size=self.scale)
    #tf.print('ReplicatedOrthogonalInitialize: tiled transposed NN depth2space shape', sampled_inits_t_nn.shape)
    sampled_inits_t = tf.transpose(sampled_inits_t_nn, perm=[1, 2, 0, 3])
    #tf.print('ReplicatedOrthogonalInitialize: tiled inits shape', sampled_inits_t.shape)
    return sampled_inits_t

  def get_config(self):
    config=super(ReplicatedOrthogonalInitialize, self).get_config()
    config['scale']=self.scale
    return config
    
def test_ReplicatedOrthogonalInitialize(shape=[3,3,16,32], scale=2):
    import matplotlib.pyplot as plt
    inits=ReplicatedOrthogonalInitialize(scale=2)(shape=shape)
    inits_3c_disp= tf.nn.depth_to_space(tf.transpose(inits, perm = [2, 0, 1, 3]), scale )[0,:,:,:3].numpy()
    inits_3c_disp_normed=(inits_3c_disp-inits_3c_disp.min())/(inits_3c_disp.max()-inits_3c_disp.min())
    plt.matshow(inits_3c_disp_normed)
    plt.show()
    return inits_3c_disp_normed

class SubpixelConv2D(tf.Module):
  """
    subpixel convolution/pixelshuffling approach (https://arxiv.org/abs/1609.05158)
    Upscaling Tensor from (any, h, w, c) to (any, h*factor, w*factor, c)
  """

  
  def __init__(self, input_dims, factor):
    """
    Args:
    input_dims : the list of dimensions of the input tensor to be upscaled
    factor : the upscaling factor to be applied
    """
    self.input_dims=input_dims
    self.factor=factor
    self.conv_output_dims = ( self.input_dims[0],
                    self.input_dims[1] * self.factor,
                    self.input_dims[2] * self.factor,
                    self.input_dims[3] // (self.factor ** 2)
                )
    self.lr_conv_channels=self.input_dims[3]*(self.factor**2)

    kernel_regul=helpers.loss.Regularizer_L1L2Ortho( l1=0.0,
                                                     l2=0.0,
                                                     ortho=0.001,
                                                     ortho_type='srip',
                                                     nb_filters=self.lr_conv_channels)
      
    #conv=None
    #if mode == '2D':
    self.lr_conv=tf.keras.layers.Conv2D(
                          filters=self.lr_conv_channels,
                          kernel_size=[3, 3],
                          strides=(1, 1),
                          padding='same',
                          data_format='channels_last',
                          dilation_rate=(1, 1),
                          activation=None,
                          use_bias=True,
                          kernel_initializer=ReplicatedOrthogonalInitialize(scale=factor),#tf.keras.initializers.Orthogonal(), #TODO, following https://arxiv.org/abs/1707.02937, copy the same ortho init for each of sub convs 
                          bias_initializer=tf.initializers.constant(0.1),
                          kernel_regularizer=kernel_regul,
                          bias_regularizer=None,
                          activity_regularizer=None,
                          kernel_constraint=None,
                          bias_constraint=None,
                          #name=name
                        )

  def __call__(self,input_features):
    """
    Args:
    input_features : the tensor of shape (h,w,c) to be upscaled
    returns : the upscaled factor to be applied of shape 
    """
    # apply conv in the initial low resolution : from (h,w,c) to (h, w, c*(factor**2)) 
    lr_features=self.lr_conv(input_features)
    #upsampling : from (h, w, c*(factor**2)) to (h*factor,w*factor,c) 
    return tf.nn.depth_to_space(lr_features, self.factor)

  def get_config(self):
      return {'input_dims':self.input_dims, 'factor':self.factor}



def atrous_Spatial_pyramid_pooling(input_features, outing_nb_features=256, rates=[1,6,12,18], kernel_sizes=[1,3,3,3]):


    features_aspp=[]
    #1. global average information
    ## global average pooling
    pooled_features=tf.math.reduce_mean(input_features,axis=[1,2], keepdims=True)
    #pooled_features=tf.keras.layers.GlobalAveragePooling2D(input_features)
    ##1x1 conv
    pooled_features_regul=helpers.loss.Regularizer_L1L2Ortho( l1=0.0,
                                                l2=0.0,#001,
                                                ortho=0.001,
                                                ortho_type='srip',
                                                nb_filters=outing_nb_features)
    pooled_features=tf.keras.layers.Conv2D(
                          filters=outing_nb_features,
                          kernel_size=[1, 1],
                          strides=(1, 1),
                          padding='same',
                          data_format='channels_last',
                          dilation_rate=(1, 1),
                          activation=None,
                          use_bias=False,
                          kernel_initializer=tf.keras.initializers.Orthogonal(),
                          bias_initializer=None,#tf.initializers.constant(0.001),
                          kernel_regularizer=pooled_features_regul,
                          bias_regularizer=None,
                          activity_regularizer=None,
                          kernel_constraint=None,
                          bias_constraint=None,
                          name='pooled_features_conv',
                        )(pooled_features)
    ## recover feature size/interpolate
    pooled_features=tf.image.resize(pooled_features,
                                         size=(input_features.shape[1],input_features.shape[2]),#(shape[0] * self.scale, shape[1] * self.scale),
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    features_aspp.append(pooled_features)
    #2. 3x3 dilated convolutions
    for rate, kernel_size in zip(rates, kernel_sizes):
        print('rate, kernel',(rate, kernel_size))
        kernel_regul=helpers.loss.Regularizer_L1L2Ortho( l1=0.0,
                                                l2=0.0,#001,
                                                ortho=0.001,
                                                ortho_type='srip',
                                                nb_filters=outing_nb_features)
        if kernel_size==1:#prefer standard conv
            features_aspp.append(tf.keras.layers.Conv2D(
                          filters=outing_nb_features,
                          kernel_size=[1, 1],
                          strides=(1, 1),
                          padding='same',
                          data_format='channels_last',
                          dilation_rate=(1, 1),
                          activation=None,
                          use_bias=False,
                          kernel_initializer=tf.keras.initializers.Orthogonal(),
                          bias_initializer=None,#tf.initializers.constant(0.001),
                          kernel_regularizer=kernel_regul,
                          bias_regularizer=None,
                          activity_regularizer=None,
                          kernel_constraint=None,
                          bias_constraint=None,
                          name='1x1_features_conv',
                        )(input_features))
        else: #prefer separable convolution for kernels>1
            features_aspp.append(tf.keras.layers.SeparableConv2D(
                        filters=outing_nb_features,
                        kernel_size=[kernel_size, kernel_size],
                        strides=(1, 1),
                        padding='same',
                        data_format='channels_last',
                        dilation_rate=(rate, rate),
                        depth_multiplier=1,
                        activation=None,
                        use_bias=True,
                        depthwise_initializer=tf.keras.initializers.he_normal(),
                        pointwise_initializer=tf.keras.initializers.Orthogonal(),
                        bias_initializer=tf.initializers.constant(0.001),
                        depthwise_regularizer=tf.keras.regularizers.l2(0.0001),
                        pointwise_regularizer=kernel_regul,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        bias_constraint=None,
                        name='ASPP_atrous'+str(rate),
                        )(input_features)
        )
    aspp_features=tf.keras.layers.Concatenate(axis=-1)(features_aspp)
    aspp_features=tf.keras.activations.relu(aspp_features)

    #fusion and compression layer
    kernel_regul_asppout=helpers.loss.Regularizer_L1L2Ortho( l1=0.0,
                                                l2=0.0,
                                                ortho=0.001,
                                                ortho_type='srip',
                                                nb_filters=outing_nb_features)
    aspp_features=tf.keras.layers.Conv2D(
                        filters=outing_nb_features,
                        kernel_size=[1, 1],
                        strides=(1, 1),
                        padding='same',
                        data_format='channels_last',
                        dilation_rate=(1, 1),
                        activation=None,
                        use_bias=True,
                        kernel_initializer=tf.keras.initializers.Orthogonal(),
                        bias_initializer=tf.initializers.constant(0.001),
                        kernel_regularizer=kernel_regul_asppout,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        name='ASPP_atrous_out',
                    )(aspp_features)
    aspp_features=tf.keras.activations.relu(aspp_features)
    return aspp_features

#CONCRETE DROPOUT from Yarin Gal & al : https://arxiv.org/abs/1705.07832
class ConcreteDropout(layers.Wrapper):
    """This wrapper allows to learn the dropout probability
        for any given input layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$
             (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and
             N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eucledian
            loss.

    # Warning
        You must import the actual layer class from tf layers,
         else this will not work.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = (np.log(init_min) - np.log(1. - init_min))
        self.init_max = (np.log(init_max) - np.log(1. - init_max))

    def build(self, input_shape=None):
        self.input_spec = layers.InputSpec(shape=input_shape)
        if hasattr(self.layer, 'built') and not self.layer.built:
            self.layer.build(input_shape)

        # initialise p
        self.p_logit = self.add_variable(name='p_logit',
                                         shape=(1,),
                                         initializer=tf.initializers.random_uniform(
                                             minval=self.init_min,
                                             maxval=self.init_max,
                                             dtype=tf.float32),
                                         dtype=tf.float32,
                                         trainable=True)
        self.p = tf.nn.sigmoid(self.p_logit[0], name='dropout_rate')
        tf.summary.scalar('dropoutRate', self.p)
        tf.add_to_collection("LAYER_P", self.p)

        # initialise regulariser / prior KL term
        input_dim = int(np.prod(input_shape[1:]))

        weight = self.layer.kernel
        with tf.name_scope('dropout_regularizer'):
            kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(
                weight)) / (1. - self.p)
            dropout_regularizer = self.p * tf.log(self.p)
            dropout_regularizer += (1. - self.p) * tf.log(1. - self.p)
            dropout_regularizer *= self.dropout_regularizer * input_dim
            regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
            # Add the regularisation loss to collection.
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        with tf.name_scope('dropout_on_input'):
            eps = 1e-7
            temp = 0.1

            unif_noise = tf.random_uniform(shape=tf.shape(x))
            drop_prob = (
                tf.log(self.p + eps)
                - tf.log(1. - self.p + eps)
                + tf.log(unif_noise + eps)
                - tf.log(1. - unif_noise + eps)
            )
            drop_prob = tf.nn.sigmoid(drop_prob / temp)
            random_tensor = 1. - drop_prob

            retain_prob = 1. - self.p
            x *= random_tensor
            x /= retain_prob
        return x

    def call(self, inputs, training=None):
        #if training:
        return self.layer.call(self.concrete_dropout(inputs))
        #else:
        #    return self.layer.call(inputs)


def concrete_dropout(inputs, layer,
                     trainable=True,
                     weight_regularizer=1e-6,
                     dropout_regularizer=1e-5,
                     init_min=0.1, init_max=0.1,
                     training=True,
                     name=None,
                     **kwargs):

    cd_layer = ConcreteDropout(layer, weight_regularizer=weight_regularizer,
                               dropout_regularizer=dropout_regularizer,
                               init_min=init_min, init_max=init_max,
                               trainable=trainable,
                               name=name)
    return cd_layer.apply(inputs, training=training)



def track_gradients(loss):
  ''' Helper function to use in the getOptimizer function
      to compute and gradients and log them into the Tensorboard
      Args:
        loss: the loss to be appled for gradients computation
      Returns:
        tvars: the trainable variables
        raw_grads: the raw gradient values
        gradient_norm: the gradient global norm
  '''
  #get all trained variables that will be optimized
  tvars = tf.trainable_variables()
  #compute gradients and track them
  raw_grads = tf.gradients(loss, tvars)
  #track gradient global norm
  gradient_norm=tf.global_norm(raw_grads, name='loss_grad_global_norm')
  tf.summary.scalar('gradient_global_norm_raw', gradient_norm)
  for grad in raw_grads:
    if grad is not None:
      trainable_nb_values=np.prod(grad.get_shape().as_list())
      if trainable_nb_values>1:
          tf.summary.histogram(grad.op.name, grad)
  return tvars, raw_grads, gradient_norm
