import tensorflow as tf
import numpy as np
from helpers.model_tcn import TCN
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

    tcn_nb_filters=64
    tcn_backbone=TCN(   nb_filters=tcn_nb_filters,
                        kernel_size=2,
                        nb_stacks=1,
                        dilations=(1, 2, 4, 8, 16, 32),
                        padding='causal',
                        use_skip_connections=False,
                        dropout_rate=0.0,
                        return_sequences=True,
                        activation='relu',
                        kernel_initializer='he_normal',
                        use_batch_norm=False,
                        use_layer_norm=False)
    m = tcn_backbone(ts)
    #keep only the sequence not directly impacted by zero padding
    m=tf.slice(m, begin=[0,tcn_backbone.receptive_field,0], size=[-1,usersettings.hparams['tsLengthIn']-tcn_backbone.receptive_field, tcn_nb_filters])
    print('m.shape',m)
    message='TCN receptive field {tcn_rf} vs input sequence length {in_l}'.format(tcn_rf=tcn_backbone.receptive_field,
                                                                                        in_l=usersettings.hparams['tsLengthIn'])
    if tcn_backbone.receptive_field >usersettings.hparams['tsLengthIn']:
        raise ValueError('TCN receptive field is wider than the input sequence lenght, fix setup: '+message)
    elif tcn_backbone.receptive_field !=usersettings.hparams['tsLengthIn']:
        print('WARNING: ',message)
    m=tf.keras.layers.Flatten()(m)    
    m=tf.keras.layers.Dense(units=output_shape[0]*output_shape[1],
                    activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.001))(m)
    pred=tf.reshape(m, shape=[-1]+list(output_shape))

    #model = Model(inputs=[i], outputs=[m])
    myModel=tf.keras.Model(inputs=[ts, yesterday_isfree, today_isfree, tomorrow_isfree], outputs=[pred])

    return myModel
