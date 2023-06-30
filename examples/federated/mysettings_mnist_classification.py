'''
@author: Alexandre Benoit & Mickaël Bettinelli, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
==> application : a personal MNIST dataset designed for clustered federated learning

FULL PROCESS USE EXAMPLE:
1. TRAIN/VAL : 

1.1 FEDERATED LEARNING
start the decentralized learning
apptainer run --nv tf2_addons.sif -m deeplearningtools.start_federated_server --usersettings=examples/federated/mysettings_image_classification.py

-> once 1.1 or 1.2 is ran, check for and use in the following steps the resulting folders, say for example one being /abs/path/to/deeplearningtools/experiments/examples/federated/my_trials_learningRate0.001_nbEpoch15_dataAugmentFalse_dropout0.2_imgHeight150_imgWidth150_2023-04-03--22:05:36/

2. SERVE ONE OF THE RESULTING MODEL : start a tensorflow model server on the produced eperiment models using command (the -psi command permits to start tensorflow model server installed in a singularity/apptainer container, here tf_server.sif):
python3 -m deeplearningtools.start_model_serving --model_dir /abs/path/to/deeplearningtools/experiments/examples/cats_dogs_classification/my_trials_learningRate0.001_nbEpoch15_dataAugmentFalse_dropout0.2_imgHeight150_imgWidth150_2023-04-03--22:05:36/ -psi /abs/path/to/tf_server.sif 

3. REQUEST MODEL : start a client that sends continuous requests to the server making use of a connected webcam
apptainer run --nv tf2_addons.sif -m deeplearningtools.experiments_manager --predict_stream=-1 --model_dir /abs/path/to/deeplearningtools/experiments/examples/federated/my_trials_learningRate0.001_nbEpoch15_dataAugmentFalse_dropout0.2_imgHeight150_imgWidth150_2023-04-03--22:05:36

Check training logs : apptainer exec --nv tf2_addons.sif tensorboard --logdir experiments/examples/federated/my_trials_learningRate0.001_nbEpoch15_dataAugmentFalse_dropout0.2_imgHeight150_imgWidth150_2023-04-03--22:05:36
'''

import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from deeplearningtools.datasets.load_federated_mnist import maybe_download_data

#-> set here your own working folder
workingFolder='experiments/examples/federated/mnist/'

#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='my_trials'

''' define here some hyperparameters to adjust the experiment
===> Note that this dictionnary will complete the session name
'''
hparams={
         'federated':'FedAvg',#set '' if not making use of federated learning or set the flower strategy name of a custom one from deeplearningtools.helpers
         'minCl':10,#minimum number of clients to allow for federated learning
         'minFit':3,#minimum number of clients to allow for a federated learning fitting round
         'clusteringMethod':'SpectralClustering', #choose among available options, currently SpectralClustering, MADC, EDC
         'learningRate':0.001,
         'nbEpoch':15,
         'procID':0, #index of learning client in the federated learning setup, may be automatically overloaded on the next few lines...
         'clusteringMethod':'SpectralClustering', #choose among available options, currently SpectralClustering, MADC, EDC
         'dropout':0.2 #used in the model definition 0.0 mean, no unit is dropped out (all data is kept)
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
used_gpu_IDs=[]

#activate federated learning if the 'federated' key appears in the hparams dictionnary 
enable_federated_learning=False
if len(hparams['federated'])>0:
    enable_federated_learning=True

#activate XLA graph optimisation, if True, GPU AND CPU XLA is applied
useXLA=False

#the metric monitored to enable early stopping and learning rate decay
monitored_loss_name='sparse_categorical_crossentropy'

#profile some training steps to check pipeline processing time bottlenecks (from Tensorboard)
use_profiling=True

# define here the used model under variable 'model'
model_file='examples/federated/model_mnist_classif.py'

# activate weight moving averaging over itarations (Polyak-Ruppert)
weights_moving_averages=True

# random seed used to init weights, etc. Use an integer value to make experiments reproducible
random_seed=42

# stop condition, taking into account if val_loss does not decrease for early_stopping_patience epoch
nbEpoch=hparams['nbEpoch']
early_stopping_patience=10

# add here any additionnal callback to use along the train/val process
addon_callbacks=[]

#set here paths to your data used for train, val -> only for config1
# -> the hyperparameter hparams['procID'] (i.e. federated client ID) is used to select the right data
maybe_download_data()
raw_data_dir_train = os.path.join(os.path.expanduser("~"),'.keras/datasets/federated_mnist/', 'config1/client' + str(hparams['procID']) + ".csv")
raw_data_dir_val = os.path.join(os.path.expanduser("~"),'.keras/datasets/federated_mnist')

raw_data_filename_extension=''
nb_train_samples=2000 #manually adjust here the number of temporal items out of the temporal block size
nb_val_samples=1000
batch_size=32
steps_per_epoch=nb_train_samples//batch_size
validation_steps=nb_val_samples//batch_size
reference_labels=['category']

########## MODEL SERVING/PRODUCTION PARAMETERS SECTION ################
#-> server and port number and other parameters to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=1#timeout limit when a client requests a served model
serve_on_gpu=True #uncomment to activate model serving on GPU instead of CPU
served_input_names=['input']
served_head_names=['category']


########## TRAIN/VAL PERSONNALIZED FUNCTIONS SECTION ################
# add here any additionnal callback to use along the train/val process
def addon_callbacks(model, train_samples, val_samples):
  ''' optionnal callbacks can be defined here
  Arg: the defined model
  Returns a list of tf.keras.callbacks or an empty list
  '''
  # Note this link to add pr_curves : https://medium.com/@akionakas/precision-recall-curve-with-keras-cd92647685e1

  return []

def get_learningRate():
  ''' define here the learning rate
  Returns a sclalar (float) or a scheduler
  '''

  return hparams['learningRate']


def get_optimizer(model, loss, learning_rate):
    '''define here the specific optimizer to be used
    Returns a tensorflow optimizer object or a string defined in Tensorflow that targets a specific optimizer with default config
    '''

    optimizer=tf.keras.optimizers.Adam(0.001)

    return optimizer

def get_metrics(model, loss):
	return [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseCategoricalCrossentropy()]

def get_total_loss(model):#inputs, model_outputs_dict, labels, weights_loss):
    '''a specific loss can be defined here or simply use a string that refers to a keras loss
    Args:
        model: the model to be optimized that may be used to focus loss on a set of specific layers or so
    Returns:
        a keras implemented loss represented by a string or a custom loss
        => it is recommended to return a tensor named 'loss' in order to enable some
        useful default options such as early stopping
    '''
    reconstruction_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return reconstruction_loss

def read_data(path, dropna=True):
    dataframe = pd.read_csv(path)
    if dropna:
        dataframe.dropna(inplace=True)
    return dataframe

def load_dataframes(path, dropna=True) -> list:
    if path.endswith('.csv') and os.path.isfile(path):
        return read_data(path, dropna=dropna)
    else:
        return []

'''Define here the input pipelines :
-1. a common function for train and validation modes
-2. a specific one for the serving model_extra_update_ops
'''
def get_input_pipeline(raw_data_files_folder, isTraining, batch_size, nbEpoch):
    ''' define an input pipeline a basic example here:
    -> load a standard dataset with tuples (image, label)
    @param raw_data_files_folder : the variable that could target a dataset/folder...
    @param isTraining : a boolean that indicates if this function is called to prepare the training or validation data pipeline
    '''
    print('Creating data pipeline, loading data from ', raw_data_files_folder, ' is training mode : ', isTraining)
    if isTraining:
        dataframe = load_dataframes(raw_data_files_folder)
        features = dataframe.iloc[:,1:]
        targets = dataframe.iloc[:,:1]
    else:
        # a single test dataset is used here for all clients and the federated server
        # -> TODO, personnalize this behavior if required
        dataframe = load_dataframes(raw_data_files_folder + '/mnist_test.csv')

        '''print("\n")
        print(dataframe)
        print(raw_data_files_folder)
        print("\n")
        '''
        features = dataframe.iloc[:,1:]
        targets = dataframe.iloc[:,:1]

    dataset = tf.data.Dataset.from_tensor_slices((features, targets))

    return dataset.batch(batch_size, drop_remainder=True).prefetch(1)

'''
################################################################################
## Serving (production) section, define here :
-get_served_module():  define how the model will be applied on production data, applying custom pre and post processing
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
'''
def get_served_module(model, model_name):
    ''' following https://www.tensorflow.org/guide/saved_model
        Create a custom module to specify how the model will be used in production/serving
        specific preprocessing can be defined as well as post-processing
    '''
    class ExportedModule(tf.Module):
        def __init__(self, model):
            super().__init__()
            self.model=model

        @tf.function(input_signature=[tf.TensorSpec(shape=[1,28, 28, 3], dtype=tf.uint8, name=served_input_names[0])])
        def served_model(self, input):
            ''' a decorated function that specifies the input data format, processing and output dict
            Args: input tensor(s)
            Returns a dictionnary of {'output key':tensor}
            '''
            pred=model(tf.cast(input, tf.float32))
            return {served_head_names[0]:tf.argmax(pred, axis=-1)}
    return ExportedModule(model)


class Client_IO:
    ''' A specific class dedicated to clients that need to interract with
    a Tensorflow server that runs the above model
    --> must have the following methods:
    def __init__(self, clientInitSpecs, debugMode): constructor that receives a debug flag
    def getInputData(self, idx): that generates data to send to the server
    def decodeResponse(self, result): that receives the response
    def finalize(self): the method call at the end of the process
    '''
    def __init__(self, clientInitSpecs={}, debugMode=False):
        ''' constructor
            Args:
               clientInitSpecs: a dictionnary to setup the client is necessary
               debugMode: set True if some debug messages should be displayed
        '''
        import cv2 #for ClientIO only

        self.debugMode=debugMode
        if self.debugMode is True:
            print('RPC Client ready to interract with the server')

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, sharex=False, sharey=False)

        #setup webcam and read a first frame (may init the camera)
        self.video_capture=cv2.VideoCapture(0)
        self.read_frame()

    def read_frame(self):
      ''' Reads a frame from the video stream
          Returns the read frame
          Raises ValueError if no frame available
      '''

      frame_is_ok, frame = self.video_capture.read()
      if not(frame_is_ok):
        raise ValueError('No input image available')
      return frame

    def getInputData(self, idx):
        ''' method that returns data samples complying with the placeholder
        defined in function get_input_pipeline_serving
        Args:
           idx: the input data index
        Returns:
           the data sample with shape and type complying with the server input
        '''
        #here, capture a frame from the webcam
        
        frame = cv2.resize(self.read_frame(), (hparams['imgHeight'], hparams['imgWidth']))

        self.frame = np.expand_dims(frame, 0)
        return {served_input_names[0]:self.frame}

    def decodeResponse(self, result):
        ''' receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        '''
        response = np.array(result.outputs[served_head_names[0]].float_val)[0]
        #print('response=', response)
        if self.debugMode is True:
            print("olala")
            #print('server model output=',result.outputs)
            #print('request shape='+str(self.frame.shape))
            #print('Answer shape='+str(response.shape))
        self.ax1.cla()
        self.ax2.cla()
        self.ax1.imshow(self.frame[0])
        #self.ax1.title('Input image')
        objects = ('Cat', 'Dog')
        y_pos = np.arange(len(objects))
        self.ax2.bar(y_pos, np.array([response, 1.0-response]), tick_label=objects)#, align='center', alpha=0.5)
        #self.ax2.set_xticklabels(['Cat', 'Dog'])
        self.ax2.set_title('Class probabilities')
        plt.pause(0.1)

    def finalize(self):
        ''' a function called when the prediction loop ends '''
        #print('Prediction process ended successfuly')
