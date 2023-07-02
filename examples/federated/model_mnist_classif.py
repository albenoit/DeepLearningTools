from tensorflow.keras import layers, metrics
from tensorflow.keras.models import Model
from sklearn.linear_model import SGDClassifier
import tensorflow as tf
import numpy as np

# singularity run ./install/tf2_addons.sif -m deeplearningtools.start_federated_server -u examples/mnist_federated/mysettings_image_classification.py  -agr ListicCFL_strategy -sim -minCl 5 -minFit 5 -rounds 5
# singularity run ./install/tf2_addons.sif -m deeplearningtools.start_federated_server -u examples/mnist_federated/mysettings_image_classification.py -sim

def model(usersettings):
    
    if usersettings.hparams['model'] == 'cnn':
      #below is proposed a classical convolutional neural network
      x=tf.keras.layers.Input(shape=(28,28,1))
      h=tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
      h=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(h)
      h=tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(h)
      h=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(h)
      h=tf.keras.layers.Flatten()(h)
      h=tf.keras.layers.Dropout(usersettings.hparams['dropout'])(h)
      h=tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(h)
      p=tf.keras.layers.Dense(units=10, activation='softmax')(h)
      my_model= tf.keras.Model(inputs=x, outputs=p)
      
    else:
      #below is proposed a classical fully connected neural network
      my_model = tf.keras.Sequential([
          tf.keras.layers.Flatten(input_shape=(784, 1)), # input layer
          
          # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
          # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
          tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)), # 1st hidden layer
          tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)), # 2nd hidden layer
          tf.keras.layers.Dropout(usersettings.hparams['dropout']), # dropout layer
          # the final layer is no different, we just make sure to activate it with softmax
          tf.keras.layers.Dense(10, activation='softmax') # output layer
      ])
    return my_model

