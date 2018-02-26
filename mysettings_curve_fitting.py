'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#-> set here your own working folder
workingFolder='experiments/curves_fitting'

#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=1#timeout limit when a client requests a served model

'''if save_model_variables_to_pandas=True, then force to save all model variables to a pandas dataframe file named 'model_parameters.bz2'
To load them later, do (update the path to your experiment):
import pandas
a=pandas.read_pickle('experiments/curves_fitting/my_test_2018-02-12--17:48:17/model_parameters.bz2')
'''
save_model_variables_to_pandas=True

#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='my_test'

#-> allow X window displays (for image and graph display purpose)
allow_display=True

#-> activate session profiling to observe ressource use and timings
do_trace_computation=True

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

#-> define here the used model under variable 'model'
#model_file='model_densenet.py'
model_file='model_curve_fitting.py'
field_of_view=20#unused
display_model_layers_info=False
#-> define here a string name used for the train, eval and served models
input_data_name='input'
model_head_embedding_name='prediction'
model_head_prediction_name=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY#'prediction'
#->define here the output that will be provided by tensorflow-server
served_head=model_head_prediction_name

#-> define the training strategy depending on the computing architecture
#---> "continuous_train_and_eval"-> single machine
#---> "train_and_evaluate" -> multiple machines/distributed training/evaluation
train_val_schedule_strategy="continuous_train_and_eval"

#-> set the number of summaries store per training epoch (more=more precise BUT higer cost)
nb_summary_per_train_epoch=4

#define image patches extraction parameters
patchSize=224

# learning rate decaying parameters
nbEpoch=300
weights_weight_decay=0.0001
initial_learning_rate=0.1
num_epochs_per_decay=150 #number of epoch keepng the same learning rate
learning_rate_decay_factor=0.1 #factor applied to current learning rate when NUM_EPOCHS_PER_DECAY is reached
predict_using_smoothed_parameters=False#set True to use trained parameters values smoothed (EMA) along the training steps (better results expected BUT STILL DOES NOT WORK WELL IN THIS CODE VERSION)

#set here paths to your data used for train, val
#-> a first set of data
raw_data_dir_train = None
raw_data_dir_val = None
raw_data_filename_extension=None
nb_train_samples=1000 #manually adjust here the number of temporal items out of the temporal block size
nb_test_samples=1000
batch_size=200
nb_classes=2
reference_labels=['values']
sigma=1.0

def target_curve(x):
    ''' the function y=f(x) to learn
    Args:
       x: input values in the form of numpy array or tensorflow Tensors
    Return:
       y=f(x)
    '''
    y=x*2+5 #main model (Numpy and Tensorflow syntax compliant)

    #add noise and adapt to the context (Numpy or Tensorflow)
    print('x='+str(x))
    if isinstance(x,tf.Tensor):
        noise=tf.random_normal(
                            shape=tf.shape(x),
                            mean=0.0,
                            stddev=1.0,)
    elif isinstance(x,np.ndarray):
        noise=np.random.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float32)
    else:
        raise ValueError('Unsupported data type')

    return y+sigma*noise

####################################################
## Define here use case specific metrics, loss, etc.
#with tf.name_scope("loss"):
def data_preprocess(features, model_placement):
    ''' define here the chosen data preprocessing that will be applied
    all the time, for training, validation and serving
    Manually specify here on which device this preprocessing should be done.
    For convenience, the placement of the model that follows this step is also provided
    so that you may want to place it on the same device.
    Args:
        features: the input data that is being processed
        model_placement: the device where the following model will be placed
    Returns:
       the preprocessed data
    '''
    # no preprocessing
    return features

def model_outputs_postprocessing_for_serving(model_outputs_dict):
    ''' define here the post-processings to be applied to each of the model outputs when used withtensorflow serving
        WARNING, in case of multiple outputs, ONE of them must be named as the
        default serving output: tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    Args:
        model_outputs_dict: the original model outputs dictionary
    Returns:
       the postprocessed outputs dictionnary
    '''
    #in this use case, we have two outputs:
    #->  code that is kept as is
    #->  semantic map logits from which we extract the most probable class index for each pixel
    postprocessed_outputs={model_head_embedding_name:model_outputs_dict['code'],
                           model_head_prediction_name:model_outputs_dict['prediction'],
                           }
    return postprocessed_outputs

def getOptimizer(loss, learning_rate, global_step):
    '''define here the specific optimizer to be used
    '''
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

def get_total_loss(inputs, model_outputs_dict, labels, weights_loss):
    '''a specific loss for data reconstruction when dealing with autoencoders
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
        weights_loss: the model weights loss that may be used for regularization
    '''
    reconstruction_loss=tf.losses.mean_squared_error(
                                model_outputs_dict['prediction'],
                                inputs,
                                weights=1.0,
                                scope=None,
                                loss_collection=tf.GraphKeys.LOSSES,
                                #reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
                                )

    return reconstruction_loss#+weights_weight_decay*weights_loss

def get_eval_metric_ops(inputs, model_outputs_dict, labels):
    '''Return a dict of the evaluation Ops.
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
    Returns:
        Dict of metric results keyed by name.
    '''
    return {
            'MSE': tf.metrics.mean_squared_error(
                labels=inputs,
                predictions=model_outputs_dict['prediction'],
                name='mean_squared_error'),
            }

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
            linear_data=target_curve(sampled_x)
            print('input sample='+str(linear_data))
        return linear_data, linear_data
    return input_fn, None

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
        serialized_tf_example, {input_data_name: serialized_tf_example})

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
        self.x=np.random.uniform(low=-10, high=10, size=[batch_size,1]).astype(np.float32)
        self.sample=target_curve(self.x)
        if self.debugMode is True:
            print('Generating input features (random values) of shape '+str(sample.shape))
        return self.sample


    def decodeResponse(self, result):
        ''' receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        '''
        response = np.array(result.outputs[served_head].float_val)
        print('request shape='+str(self.sample.shape))
        print('Answer shape='+str(response.shape))
        self.ax.cla()
        self.ax.plot(self.x, self.sample,'r+')
        self.ax.plot(self.x, response, 'b+')
        plt.pause(1)

    def finalize(self):
        ''' a function called when the prediction loop ends '''
        print('Prediction process ended successfuly')
