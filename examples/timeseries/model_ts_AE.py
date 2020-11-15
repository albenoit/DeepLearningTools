import tensorflow as tf
import numpy as np

def model(usersettings):

    input_shape=(usersettings.hparams['tsLengthIn'], usersettings.hparams['nbChannels'] )
    output_shape=(usersettings.hparams['tsLengthOut'], usersettings.hparams['nbChannels'])
    print('model input=',input_shape)
    print('model output=',output_shape)
    # Define the model
    ts = tf.keras.layers.Input(shape=input_shape, name='input_time_series')
    yesterday_isfree = tf.keras.layers.Input(shape=(1,), name='input_yesterday_isfree')
    today_isfree=tf.keras.layers.Input(shape=(1,), name='input_today_isfree')
    tomorrow_isfree=tf.keras.layers.Input(shape=(1,), name='input_tomorrow_isfree')


    #2 conv layers at initial scale
    h_ts=tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(3,),
                           kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                           activation='relu')(ts)
    print('model input=',h_ts)

    h_ts=tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(3,),
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           activation='relu')(h_ts)
    print('model input=',h_ts)
    #pool/subsample
    h_ts=tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(h_ts)
    print('model input=',h_ts)
 
    #2 new conv layers
    h_ts=tf.keras.layers.Conv1D(filters=64,
                           kernel_size=(3,),
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           activation='relu')(h_ts)
    print('model input=',h_ts)
    h_ts=tf.keras.layers.Conv1D(filters=64,
                           kernel_size=(3,),
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           activation='relu')(h_ts)
    print('model input=',h_ts)
    h_ts=tf.keras.layers.Flatten()(h_ts)
    print('model input=',h_ts)

    #manage metadata
    h_meta=tf.keras.layers.Concatenate(axis=-1)([yesterday_isfree, today_isfree, tomorrow_isfree])
    h=tf.keras.layers.Dense(units=10, kernel_regularizer=tf.keras.regularizers.l2(0.001))(h_meta)
    
    #fuse all features together
    h=tf.keras.layers.Concatenate(axis=-1)([h_ts, h_meta])
    '''h=tf.keras.layers.Dense(units=100,
                    activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(h)
    '''
    delta=tf.keras.layers.Dense(units=output_shape[0]*output_shape[1],
                    activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.001))(h)
    delta=tf.reshape(delta, shape=[-1]+list(output_shape))

    print('model delta=',delta)

    #the model only predicts the delta from the last input data
    ts_in_last=tf.slice(ts, begin=[0,usersettings.hparams['tsLengthOut']-1,0], size=[-1,1,usersettings.hparams['nbChannels']])
    ts_in_last=tf.tile(ts_in_last, [1,usersettings.hparams['tsLengthOut'],1])
    print('model ts_in_last=',ts_in_last)

    pred=ts_in_last+delta
    print('model delta=',delta)
    myModel=tf.keras.Model(inputs=[ts, yesterday_isfree, today_isfree, tomorrow_isfree], outputs=[pred])

    return myModel
