import tensorflow as tf

def model(usersettings):

    input_dim=1
    x=tf.keras.Input(shape=[input_dim], name='input')

    h=tf.keras.layers.Dense(units=usersettings.hparams['hiddenNeurons'],
                    activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)#tf.keras.regularizers.l2(0.01))(data)
    h=tf.keras.layers.Dense(units=usersettings.hparams['hiddenNeurons'],
                    activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)#tf.keras.regularizers.l2(0.01))(data)
    pred=tf.keras.layers.Dense(units=input_dim,
                    activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01))(h)
    
    myModel=tf.keras.Model(inputs=x, outputs=[pred])
    return myModel
