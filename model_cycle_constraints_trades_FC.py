import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn import ModeKeys

# =============================== GLOBAL MODEL ====================================
#building Y=F(X)
def F(X, reuse_params=False):
    ''' Y=F(X)
    Args:
       X: input Tensor
       reuse_params: set True to create new Variables and False to reuse existing ones
    Returns:
       tensor Y=F(X)
    Raises:
         Error if reuse_params is set to True but variable does not exist
    '''
    h_dim=200
    print('F(X) with X='+str(X))
    X_flat=tf.layers.flatten(X)
    print('F(X) with flatten(X)='+str(X))
    X_dim=X.get_shape().as_list()
    with tf.variable_scope('F', reuse=reuse_params):
        h=tf.layers.Dense(units=h_dim,
                        activation=tf.nn.relu)(X_flat)
        h=tf.layers.Dense(units=h_dim,
                        activation=tf.nn.relu)(h)
        h=tf.layers.Dense(units=h_dim,
                        activation=tf.nn.relu)(h)
        pred_y=tf.layers.Dense(units=X.get_shape().as_list()[2],
                        activation=None)(h)#define parameters
    return tf.reshape(pred_y,X_dim)

#building X=G(Y)
def G(Y, reuse_params=False):
    ''' Y=F(X)
    Args:
       Y: input Tensor
       reuse_params: set True to create new Variables and False to reuse existing ones
    Returns:
       tensor Y=F(X)
    Raises:
         Error if reuse_params is set to True but variable does not exist
    '''
    h_dim=200
    print('G(Y) with Y='+str(Y))
    Y_flat=tf.layers.flatten(Y)
    print('G(Y) with flatten(Y)='+str(Y_flat))
    Y_dim=Y.get_shape().as_list()
    with tf.variable_scope('G', reuse=reuse_params):
        h=tf.layers.Dense(units=h_dim,
                        activation=tf.nn.relu)(Y_flat)
        h=tf.layers.Dense(units=h_dim,
                        activation=tf.nn.relu)(h)
        h=tf.layers.Dense(units=h_dim,
                        activation=tf.nn.relu)(h)
        pred_x=tf.layers.Dense(units=Y.get_shape().as_list()[2],
                        activation=None)(h)#define parameters
    return tf.reshape(pred_x, Y_dim)

def model(data,
            n_outputs,
            hparams,
            mode):

    #get input data dim
    data_initial_shape=data.get_shape().as_list()
    print('Model input data shape='+str(data_initial_shape))
    if len(data_initial_shape)!=4:
        raise ValueError('Expecting a time series batch of shape [batch_size, measures]')

    time_series_length=data_initial_shape[2]/2
    if mode != ModeKeys.INFER: #for TRAIN and VAL modes, 2 successive time series are provided
        X=tf.slice(data, begin=[0,0,0,0], size=[-1,-1,time_series_length,-1])
        Y=tf.slice(data, begin=[0,0,time_series_length,0], size=[-1,-1,time_series_length,-1])
        X=tf.Print(X,[X,Y], message="[X,Y]=")
    else: #in prediction mode (ModeKeys.INFER), a single time series predicts past and future
        X=data
        Y=data

    #defining singular functions
    y_est=F(X, False)
    x_est=G(Y, False)
    #defining cycle functions
    x_cycle=G(y_est, True)
    y_cycle=F(x_est, True)

    return {'F_x':y_est, 'G_y':x_est, 'GoF_x':x_cycle, 'FoG_y':y_cycle}
