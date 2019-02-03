# python 2&3 compatibility management
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import tensorflow as tf
import numpy as np

def weight_variable(shape):
    '''MSRA initialization of a given weigths tensor
    @param shape, the 4d tensor shape
    variable is allocated on the CPU memory even if processing will use it on GPU
    '''
    with tf.device('/cpu:0'):
        n= np.prod(shape[:3])#n_input_channels*kernelShape
        trunc_stddev = np.sqrt(1.3 * 2.0 / n)
        initial = tf.truncated_normal(shape, 0.0, trunc_stddev)
        weights=tf.get_variable(name='weights', initializer=initial)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
        return weights

def bias_variable(shape):
    ''' basic constant bias variable init (a little above 0)
    @param shape, the 4d tensor shape
    variable is allocated on the CPU memory even if processing will use it on GPU
    '''
    with tf.device('/cpu:0'):
        initial = tf.constant(0.01, shape=shape)
        return tf.get_variable(name='biases', initializer=initial)

# =============================== GLOBAL MODEL ====================================

def model(data,
            hparams,
            mode):

    #get input data dim
    data_initial_shape=data.get_shape().as_list()
    print('Model input data shape='+str(data_initial_shape))
    X_dim=data.get_shape().as_list()[-1]

    h_dim=hparams.hiddenNeurons #hidden layer size

    h=tf.layers.Dense(units=h_dim,
                    activation=tf.nn.relu)(data)
    pred=tf.layers.Dense(units=X_dim,
                    activation=None)(h)
    '''with tf.variable_scope('Encoder'):
        #encoder parameters and graph
        E_W1 = weight_variable([X_dim, h_dim])
        E_b1 = bias_variable(shape=[h_dim])
        h = tf.nn.relu(tf.matmul(data, E_W1) + E_b1)
        #raw_input('hidden layer (W,b)='+str((E_W1,E_b1, h)))
    with tf.variable_scope('Decoder'):
        D_W1 = weight_variable([h_dim, X_dim])
        D_b1 = bias_variable(shape=[X_dim])
        pred = tf.matmul(h, D_W1) + D_b1
        #raw_input('hidden layer (W,b, tensor)='+str((D_W1,D_b1, pred)))
    '''
    return {'code':h, 'prediction':pred}
