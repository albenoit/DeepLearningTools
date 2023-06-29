from tensorflow.keras import layers, metrics
from tensorflow.keras.models import Model
from sklearn.linear_model import SGDClassifier
import tensorflow as tf
import numpy as np

# singularity run ./install/tf2_addons.sif -m deeplearningtools.start_federated_server -u examples/mnist_federated/mysettings_image_classification.py  -agr ListicCFL_strategy -sim -minCl 5 -minFit 5 -rounds 5
# singularity run ./install/tf2_addons.sif -m deeplearningtools.start_federated_server -u examples/mnist_federated/mysettings_image_classification.py -sim

def model(usersettings):
    
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(784, 1)), # input layer
        
        # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
        # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
        tf.keras.layers.Dense(200, activation='relu'), # 1st hidden layer
        tf.keras.layers.Dense(200, activation='relu'), # 2nd hidden layer
        
        # the final layer is no different, we just make sure to activate it with softmax
        tf.keras.layers.Dense(10, activation='softmax') # output layer
    ])

    return model
