# python 2&3 compatibility management
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import tensorflow as tf
import numpy as np

from helpers_model import ConcreteDropout
from tensorflow.contrib.learn import ModeKeys

# =============================== GLOBAL MODEL ====================================
def model(data,
            hparams,
            mode):


    is_training = mode == ModeKeys.TRAIN
    #get input data dim
    data_initial_shape=data.get_shape().as_list()
    print('Model input data shape='+str(data_initial_shape))
    X_dim=data.get_shape().as_list()[-1]
    l = 1e-6#-4
    wd = l**2. / X_dim
    dd = 2. / X_dim
    h_dim=hparams.hiddenNeurons #hidden layer size
    print('Raw input dim, hdim='+str((X_dim, h_dim)))
    with tf.variable_scope('Encoder'):
        #encoder parameters and graph
        h=ConcreteDropout(tf.layers.Dense(units=200, activation=tf.nn.relu),
                    weight_regularizer=wd,
                    dropout_regularizer=dd,
                    trainable=True)(data, training=is_training)
        h=ConcreteDropout(tf.layers.Dense(units=200, activation=tf.nn.relu),
                    weight_regularizer=wd,
                    dropout_regularizer=dd,
                    trainable=True)(h, training=is_training)

    with tf.variable_scope('Decoder'):
        #standard prediction output
        pred=ConcreteDropout(tf.layers.Dense(units=X_dim,
                                    activation=None),
                                    weight_regularizer=wd,
                                    dropout_regularizer=dd,
                                    trainable=True)(h, training=is_training)
        #variance prediction output
        log_var = ConcreteDropout(tf.layers.Dense(units=X_dim,
                                    activation=None),
                                    weight_regularizer=wd,
                                    dropout_regularizer=dd,
                                    trainable=True)(h, training=is_training)

    return {'code':h, 'prediction':pred, 'log_var':log_var}
