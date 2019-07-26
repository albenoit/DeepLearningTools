# python 2&3 compatibility management
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import tensorflow as tf
import numpy as np

def model(data,
            hparams,
            mode):

    #get input data dim
    data_initial_shape=data.get_shape().as_list()
    print('Model input data shape='+str(data_initial_shape))
    X_dim=data.get_shape().as_list()[-1]


    h=tf.keras.layers.Dense(units=hparams.hiddenNeurons,
                    activation='relu', kernel_regularizer=tf.nn.l2_loss)(data)#tf.keras.regularizers.l2(0.01))(data)
    pred=tf.keras.layers.Dense(units=X_dim,
                    activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01))(h)
    return {'code':h, 'prediction':pred}
