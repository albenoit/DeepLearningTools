""" a LSTM model for time series forecasting
@author : Alexandre Benoit, LISTIC Lab, FRANCE
2018

Hints and explenations on this blog : https://machinelearningmastery.com/use-weight-regularization-lstm-networks-time-series-forecasting/
"""

import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn import ModeKeys

def RNN_layer(X, n_units, num_steps, is_training, layer_name):
  with tf.variable_scope(layer_name):
    '''basic_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=n_units, activation=tf.nn.relu)
    rnn_output, states=tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
    return rnn_output
    '''

    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_units,
                                        state_is_tuple=True,
                                        activation=tf.tanh)
    initial_state = last_states = cell.zero_state(X.get_shape().as_list()[0], tf.float32)
    #raw_input('LSTM cell input='+str(X))
    dropoutrate=1.0
    if is_training is True and X.get_shape().as_list()[2]>1 :
      dropoutrate=0.8

    '''cell=tf.nn.rnn_cell.DropoutWrapper(cell,
                                       input_size=X.get_shape().as_list()[2],
                                       input_keep_prob=dropoutrate,
                                       output_keep_prob=1.0,
                                       state_keep_prob=1.0,#dropoutrate,
                                       variational_recurrent=True,#if True, the same dropout mask is applied at every step, as described in Y. Gal, Z Ghahramani. "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks". https://arxiv.org/abs/1512.05287
                                       dtype=tf.float32)
    '''
    outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float32,
        initial_state=initial_state,
        #sequence_length=num_steps,
        inputs=X)
    return outputs

def RNN(X, is_training):
  n_hidden=50
  output=1
  num_steps=X.get_shape().as_list()[1]
  rnn_outputs=RNN_layer(X, 30, num_steps, is_training, 'l1')#RNN_layer(RNN_layer(X, n_hidden),n_hidden)
  #raw_input('rnn_outputs_l1='+str(rnn_outputs))
  #rnn_outputs=tf.Print(rnn_outputs,[tf.shape(rnn_outputs)],message='l1')
  rnn_outputs=RNN_layer(rnn_outputs, 10, num_steps, is_training, 'l2')#RNN_layer(RNN_layer(X, n_hidden),n_hidden)
  #rnn_outputs=tf.Print(rnn_outputs,[tf.shape(rnn_outputs)],message='l2')

  '''
  #get the last output:
  last_rnnoutput=rnn_outputs[:,-1,:]
  #equivalent to:
  # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
  # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
  val = tf.transpose(rnn_outputs, [1, 0, 2])
  # last.get_shape() = (batch_size, lstm_size)
  last_rnnoutput = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")
  print('last_rnnoutput='+str(last_rnnoutput))
  '''
  print('rnn_outputs_l2='+str(rnn_outputs))
  rnn_out=tf.layers.flatten(rnn_outputs[:,-1,:])
  print('stacked_rnn_outputs='+str(rnn_out))
  rnn_output=tf.layers.dense(rnn_out, num_steps,
                             kernel_initializer=tf.glorot_uniform_initializer(),
                             bias_initializer=tf.constant_initializer([0.1]*num_steps),
                             kernel_regularizer=tf.nn.l2_loss)
  #raw_input('rnn_output_raw='+str(rnn_output))
  rnn_output=tf.reshape(rnn_output, [-1,num_steps, output])
  #raw_input('rnn_output='+str(rnn_output))
  return rnn_output

# =============================== GLOBAL MODEL ====================================
#building Y=F(X)
def F(X, is_training, reuse_params=False):
    ''' Y=F(X)
    Args:
       X: input Tensor
       reuse_params: set True to create new Variables and False to reuse existing ones
    Returns:
       tensor Y=F(X)
    Raises:
         Error if reuse_params is set to True but variable does not exist
    '''
    print('F(X) with X='+str(X))

    with tf.variable_scope('F', reuse=reuse_params):
        #LSTM layers parts
        lstm_output=RNN(X, is_training)
        print('F:lstm_output='+str(lstm_output))
    return lstm_output

#building X=G(Y)
def G(Y, is_training, reuse_params=False):
    ''' X=G(Y)
    Args:
       Y: input Tensor
       reuse_params: set True to create new Variables and False to reuse existing ones
    Returns:
       tensor X=G(Y)
    Raises:
         Error if reuse_params is set to True but variable does not exist
    '''
    print('G(Y) with Y='+str(Y))

    with tf.variable_scope('G', reuse=reuse_params):
        #first reverse the temporal sequence
        Y_rev=tf.reverse(
                  Y,
                  axis=[1],
                  name="reverse_Y"
                  )
        #apply the 'anticausal' network
        h=RNN(Y_rev, is_training)
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
    #defining singular functions
    y_est=F(X, is_training, False)
    x_est=G(Y, is_training, False)
    #defining cycle functions
    x_cycle=G(F(X, is_training, True), is_training, True)#replace by tf.identity(X) to explicitely eliminate this subgraph
    y_cycle=F(G(Y, is_training, True), is_training, True)#replace by tf.identity(Y) to explicitely eliminate this subgraph

    return {'F_x':y_est, 'G_y':x_est, 'GoF_x':x_cycle, 'FoG_y':y_cycle}
