import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn import ModeKeys
import model_densenet_1D_AE

#set here how tensors channels are organized, following tensorflow naming convention
nn_data_fmt='channels_last'

# Redefining CausalConv1D to simplify its return values
class CausalConv1D(tf.layers.Conv1D):
    def __init__(self, filters,
               kernel_size,
               strides=1,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )
        self.fov=(kernel_size-1)*dilation_rate

    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        print('padding(kernel={k}, dilation={d})={p}'.format(k=self.kernel_size[0], d=self.dilation_rate[0], p=padding))
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        return super(CausalConv1D, self).call(inputs)

class TemporalBlock(tf.layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2,
                 trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=None,#tf.nn.relu,
            name="conv1")
        self.conv2 = CausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=None,#tf.nn.relu,
            name="conv2")
        self.down_sample = None
        self.fov=self.conv1.fov+self.conv2.fov


    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = tf.layers.Dropout(self.dropout)#, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = tf.layers.Dropout(self.dropout)#, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        if input_shape[channel_dim] != self.n_outputs:
            self.down_sample = tf.layers.Conv1D(self.n_outputs, kernel_size=1,
                                                 activation=None, data_format="channels_last", padding="valid")
            print('Downsampling')
            #self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)

    def call(self, inputs, training=True):
        '''x = self.conv1(inputs)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        print('TemporalBlock output='+str(inputs))
        return tf.nn.relu(x + inputs)
        '''
        #x = tf.nn.relu(tf.contrib.layers.layer_norm(inputs))
        x = self.dropout1(inputs, training=True)#always active for bayesian inference
        x = self.conv1(x)
        x = tf.nn.relu(tf.contrib.layers.layer_norm(x))
        x = self.dropout2(x, training=True)#always active for bayesian inference
        x = self.conv2(x)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        print('TemporalBlock output='+str(inputs))
        return tf.nn.relu(x + inputs)

class TemporalConvNet(tf.layers.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2,
                 trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.layers = []
        num_levels = len(num_channels)
        print('Building TCN...')
        self.fov=1
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            new_Temporal_block=TemporalBlock(out_channels, kernel_size, strides=1, dilation_rate=dilation_size,
                          dropout=dropout, name="tblock_{}".format(i))
            self.fov+=new_Temporal_block.fov
            self.layers.append(new_Temporal_block)

            print('Level {level}, dilation_size={dilate}, out_channels={out}'.format(level=i,
                                                                                     dilate=dilation_size,
                                                                                     out=out_channels))
        print('fov COMPUTED='+str(self.fov))
        print('fov ESTIM=   '+str(1 + 2*(kernel_size-1)*(2**num_levels-1)))


    def call(self, inputs, training=True):
        outputs = inputs
        print('*** TCN input='+str(outputs))
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        print('*** TCN output='+str(outputs))
        lastLayer=tf.layers.Conv1D(1,
                                    kernel_size=[1],
                                    strides=1,
                                    padding='valid',
                                    data_format='channels_last',
                                    dilation_rate=1,
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=None,
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    trainable=True)(outputs)

        lastLayer=tf.layers.Dense(units=outputs.get_shape().as_list()[1],
                                  trainable=True)(tf.layers.flatten(lastLayer))

        return tf.expand_dims(lastLayer,-1)
# =============================== GLOBAL MODEL ====================================


def NN(data, is_training):
  ''' A Neural net architecture that can be reused by F(X) and G(Y) '''
  tcn = TemporalConvNet([16, 16, 16], 3, 0.25)
  tcn_out=tcn(data, is_training)
  return tcn_out

#building Y=F(X)
def F(X, reuse_params=False, is_training=True):
    ''' Y=F(X)
    Args:
       X: input Tensor
       reuse_params: set True to create new Variables and False to reuse existing ones
    Returns:
       tensor Y=F(X)
    Raises:
         Error if reuse_params is set to True but variable does not exist
    '''
    print('## F(X) with X='+str(X))

    with tf.variable_scope('F', reuse=reuse_params):
        h=NN(X, is_training)
    print('## F(X) output ='+str(h))
    return h

#building X=G(Y)
def G(Y, reuse_params=False, is_training=True):
    ''' Y=F(X)
    Args:
       Y: input Tensor
       reuse_params: set True to create new Variables and False to reuse existing ones
    Returns:
       tensor Y=F(X)
    Raises:
         Error if reuse_params is set to True but variable does not exist
    '''
    print('## G(Y) with Y='+str(Y))
    with tf.variable_scope('G', reuse=reuse_params):
        #first reverse the temporal sequence
        Y_rev=tf.reverse(
                  Y,
                  axis=[1],
                  name="reverse_Y"
                  )
        #apply the 'anticausal' network
        h=NN(Y_rev, is_training)
        #finaly get back to the original causal ordering
        h_rev=tf.reverse(
                  h,
                  axis=[1],
                  name="reverse_X_est"
                  )
    print('## G(Y) output ='+str(h_rev))
    return h_rev

def model(data,
            hparams,
            mode):

    #get input data dim
    data_initial_shape=data.get_shape().as_list()
    print('Model input data shape='+str(data_initial_shape))
    if len(data_initial_shape)!=3:
        raise ValueError('Expecting a time series batch of shape [batch_size, time_series,channels]')

    is_training=False#default value
    time_series_length=data_initial_shape[1]/2
    if mode != ModeKeys.INFER: #for TRAIN and VAL modes, 2 successive time series are provided
        if mode == ModeKeys.TRAIN:
          is_training=True
        X=tf.slice(data, begin=[0,0,0], size=[-1,time_series_length,-1])
        Y=tf.slice(data, begin=[0,time_series_length,0], size=[-1,time_series_length,-1])
        #X=tf.Print(X,[X,Y], message="[X,Y]=")
    else: #in prediction mode (ModeKeys.INFER), a single time series predicts past and future
        X=data
        Y=data
    print('model inputs: X,Y='+str((X,Y)))

    '''X=tf.squeeze(X,1)
    Y=tf.squeeze(Y,1)
    '''
    print('model inputs: SQUEEZED X,Y='+str((X,Y)))
    #defining singular functions
    y_est=F(X, False, is_training)
    x_est=G(Y, False, is_training)

    assert(y_est.get_shape().as_list() == x_est.get_shape().as_list(), '************** F and G functions should have the same output shape, F,G outputs='+str((x_est,y_est)))
    #defining cycle functions
    x_cycle=G(y_est, True, is_training)
    y_cycle=F(x_est, True, is_training)

    '''x_est=tf.expand_dims(x_est,1)
    y_est=tf.expand_dims(y_est,1)
    x_cycle=tf.expand_dims(x_cycle,1)
    y_cycle=tf.expand_dims(y_cycle,1)
    '''
    return {'F_x':y_est, 'G_y':x_est, 'GoF_x':x_cycle, 'FoG_y':y_cycle}
