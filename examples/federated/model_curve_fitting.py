import tensorflow as tf

def model(usersettings):

    input_dim=1
    x_in=tf.keras.Input(shape=[input_dim], name='input')

    if usersettings.hparams['hiddenNeurons']>0:
        h=tf.keras.layers.Dense(units=usersettings.hparams['hiddenNeurons'],
                                activation=usersettings.hparams['activation'],
                                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                bias_initializer=tf.keras.initializers.Constant(0.1),
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    else:
        h=x_in
    pred=tf.keras.layers.Dense(units=input_dim,
                               activation=None,
                               kernel_initializer=tf.keras.initializers.GlorotNormal(),
                               bias_initializer=tf.keras.initializers.Constant(0.1),
                               name='output')(h)
    myModel=tf.keras.Model(inputs=x_in, outputs=[pred])
    return myModel


