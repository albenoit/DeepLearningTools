'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
'''
# python 2&3 compatibility management
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#-> set here your own working folder
workingFolder='experiments/curve_fitting'

#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=1#timeout limit when a client requests a served model
input_data_name='input'
model_head_prediction_name=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY#'prediction'
#->define here the output that will be provided by tensorflow-server
from tensorflow.python.saved_model import signature_constants
served_head="predict"#signature_constants.REGRESS_OUTPUTS#signature_constants.REGRESS_METHOD_NAME

'''if save_model_variables_to_pandas=True, then force to save all model variables to a pandas dataframe file named 'model_parameters.bz2'
To load them later, do (update the path to your experiment):
import pandas
a=pandas.read_pickle('experiments/curves_fitting/my_test_2018-02-12--17:48:17/model_parameters.bz2')
'''
save_model_variables_to_pandas=True

display_model_layers_info=False#do not output ops and vars placement on console

#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='premade_estimator'


''' define here some hyperparameters to adjust the experiment
===> Note that this dictionnary will complete the session name
'''
hparams={'hiddenNeurons':10,#set the number of neurons per hidden layers
         }
''''set the list of GPUs involved in the process. HOWTO:
->if using CPU only mode, let an empty list
->if using a single GPU, only the first ID of the list will be considered
->if using multiple GPUs, each GPU ID will be considered
=> general recommendation: always try to focus on unused GPUs to avoid conflicts
with other processing jobs, yours and the ones of your colleagues.
Then, connect to the processing node and type in command line 'nvidia-smi'
to check which gpu is free (very few used memory and GPU )
'''
used_gpu_IDs=[0]
#set here XLA optimisation flags, either tf.OptimizerOptions.OFF#ON_1#OFF
XLA_FLAG=tf.OptimizerOptions.OFF#ON_1#OFF

premade_estimator=tf.estimator.DNNRegressor(hidden_units=[hparams['hiddenNeurons']],
                feature_columns=[tf.feature_column.numeric_column('x')],
                label_dimension=1,
                weight_column=None,
                #label_vocabulary=None,
                optimizer='Adagrad',
                activation_fn=tf.nn.relu,
                dropout=None,
                input_layer_partitioner=None,
                config=None,
                warm_start_from=None,
                loss_reduction=tf.losses.Reduction.SUM
            )
#-> set the number of summaries store per training epoch (more=more precise BUT higer cost)
nb_summary_per_train_epoch=4

#random seed used to init weights, etc. Use an integer value to make experiments reproducible
random_seed=42

# learning rate decaying parameters
nbEpoch=300
weights_weight_decay=0.0001
initial_learning_rate=0.1
num_epochs_per_decay=150 #number of epoch keepng the same learning rate
learning_rate_decay_factor=0.1 #factor applied to current learning rate when NUM_EPOCHS_PER_DECAY is reached
predict_using_smoothed_parameters=False# ONLY FOR CUSTOM MODELS (not in this demo): set True to use trained parameters values smoothed (EMA) along the training steps (better results expected)

#set here paths to your data used for train, val
#-> a first set of data
raw_data_dir_train = None
raw_data_dir_val = None
raw_data_filename_extension=None
nb_train_samples=1000 #manually adjust here the number of temporal items out of the temporal block size
nb_test_samples=1000
batch_size=200

def numpycurve(x):
    sigma=1.0
    noise=np.random.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float32)
    y=x**2
    '''x_neg=np.where(x<=0)
    x_pos=np.where(x>0)
    y[x_neg]=x[x_neg]**2
    y[x_pos]=np.sqrt(x[x_pos])*5
    '''
    return y+noise

def target_curve(x):
    ''' the function y=f(x) to learn
    Args:
       x: input values in the form of numpy array or tensorflow Tensors
    Return:
       y=f(x)
    '''
    #add noise and adapt to the context (Numpy or Tensorflow)
    #print('x='+str(x))
    if isinstance(x,tf.Tensor):
        #explicitely reshaping output to help graph construction
        y=tf.reshape(tf.py_func(numpycurve, [x], tf.float32), x.shape)
        return y

    elif isinstance(x,np.ndarray):
        return numpycurve(x)

    raise ValueError('Unsupported data type')

'''Define here the input pipelines :
-1. a common function for train and validation modes
-2. a specific one for the serving model_extra_update_ops
'''
def get_input_pipeline_train_val(batch_size, raw_data_files_folder, shuffle_batches):
    ''' define an input pipeline able to load temporal series from a set of
    CSV files and a batch size specified as inputs
    TODO, look at the doc here : https://www.tensorflow.org/programmers_guide/datasets
    @param batch_size : the expected size of a batch
    @param raw_data_files_folder : the folder where CSV files are stored
    @param shuffle_batches : a boolean that activates batch shuffling
    '''
    def input_fn():
        with tf.name_scope("generate_data"):
            # a simple uniform distribution centered on zero
            sampled_x = tf.random_uniform(shape=[batch_size,1], minval=-5, maxval=5)
            sampled_y=target_curve(sampled_x)
            print('input sample='+str(sampled_y))
        return {'x':sampled_x}, sampled_y
    return input_fn, None

def features_dict_to_tensor(features):
  data_vectors = tf.feature_column.input_layer(features, [tf.feature_column.numeric_column('x')])
  return data_vectors
'''
################################################################################
## Serving (production) section, define here :
-get_input_pipeline_serving():  the input placeholder of the server that will receive the data
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
'''
def get_input_pipeline_serving():
    '''Build the serving inputs, expecting messages made of :
    -> a batch of size 1.
    -> a data buffer of type float32 of the same shape as each of the elements used along training (no preliminary normalisation is expected)
    '''
    serialized_tf_example = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, 1],
        name='serialized_input_data')

    return tf.estimator.export.ServingInputReceiver(
        {'x':serialized_tf_example}, {input_data_name: serialized_tf_example})

class Client_IO:
    ''' A specific class dedicated to clients that need to interract with
    a Tensorflow server that runs the above model
    --> must have the following methods:
    def __init__(self, debugMode): constructor that receives a debug flag
    def getInputData(self, idx): that generates data to send to the server
    def decodeResponse(self, result): that receives the response
    '''
    def __init__(self, debugMode):
        ''' constructor
            Args:
               debugMode: set True if some debug messages should be displayed
        '''
        self.debugMode=debugMode
        if self.debugMode is True:
            print('RPC Client ready to interract with the server')

        self.fig, self.ax = plt.subplots()

    def getInputData(self, idx):
        ''' method that returns data samples complying with the placeholder
        defined in function get_input_pipeline_serving
        Args:
           idx: the input data index
        Returns:
           the data sample with shape and type complying with the server input
        '''
        #here, only random numbers
        self.x=np.random.uniform(low=-5, high=5, size=[batch_size,1]).astype(np.float32)
        self.target=target_curve(self.x)
        if self.debugMode is True:
            print('Generating input features (random values) of shape '+str(self.target.shape))
        return self.x


    def decodeResponse(self, result):
        ''' receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        '''
        response = np.array(result.outputs['predictions'].float_val)
        if self.debugMode is True:
            print('request shape='+str(self.x.shape))
            print('Answer shape='+str(response.shape))
        self.ax.cla()
        self.ax.plot(self.x, self.target,'r+')
        self.ax.plot(self.x, response, 'b+')
        plt.pause(1)

    def finalize(self):
        ''' a function called when the prediction loop ends '''
        print('Prediction process ended successfuly')
