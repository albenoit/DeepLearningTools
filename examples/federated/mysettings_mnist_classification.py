'''
@author: Alexandre Benoit & Mickaël Bettinelli, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
==> application : a personalized MNIST dataset designed for federated learning

FULL PROCESS USE EXAMPLE:
1. TRAIN/VAL : 
    
1 FEDERATED LEARNING
start the decentralized learning
apptainer run --nv tf2_addons.sif -m deeplearningtools.start_federated_server --usersettings=examples/federated/mysettings_mnist_classification.py

-> once done, check for and use in the following steps the resulting folders, say for example one being /abs/path/to/deeplearningtools/experiments/examples/federated/my_trials_learningRate0.001_nbEpoch15_dataAugmentFalse_dropout0.2_imgHeight150_imgWidth150_2023-04-03--22:05:36/

2. SERVE ONE OF THE RESULTING MODEL : start a tensorflow model server on the produced eperiment models using command (the -psi command permits to start tensorflow model server installed in a singularity/apptainer container, here tf_server.sif):
python3 -m deeplearningtools.start_model_serving --model_dir /abs/path/to/deeplearningtools/experiments/examples/cats_dogs_classification/my_trials_learningRate0.001_nbEpoch15_dataAugmentFalse_dropout0.2_imgHeight150_imgWidth150_2023-04-03--22:05:36/ -psi /abs/path/to/tf_server.sif 

3. REQUEST MODEL : start a client that sends continuous requests to the server making use of a connected webcam
apptainer run --nv tf2_addons.sif -m deeplearningtools.experiments_manager --predict_stream=-1 --model_dir /abs/path/to/deeplearningtools/experiments/examples/federated/my_trials_learningRate0.001_nbEpoch15_dataAugmentFalse_dropout0.2_imgHeight150_imgWidth150_2023-04-03--22:05:36

If you need to compare federated learning approaches with centralized learning, you can use the following options:
-> federated learning : set the 'federated' key in the following hparams dictionnary to 'FedAvg' or to the name of a custom strategy defined in Flwr or deeplearningtools.helpers.federated and use the above instructions
-> centralized learning : let the 'federated' key empty in the hparams dictionnary then, run the experiment as usual using the deeplearningtools.experiments_manager module (see the other centralized learning examples)

Check training logs : apptainer exec --nv tf2_addons.sif tensorboard --logdir experiments/examples/federated/my_trials_learningRate0.001_nbEpoch15_dataAugmentFalse_dropout0.2_imgHeight150_imgWidth150_2023-04-03--22:05:36
'''

import subprocess
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from deeplearningtools.datasets.load_federated_mnist import maybe_download_data
from deeplearningtools.DataProvider_input_pipeline import extractFilenames
from deeplearningtools.helpers import metrics
import cv2 #only used on the ClientIO side to capture and display images

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
         'minFit':2,#minimum number of clients to allow for a federated learning fitting round
         'learningRate':0.001,
         'nbEpoch':1,#sets either the number of epoch per cleint for each federated round OR sets the total number of epoch for centralised learning
         'procID':0, #index of learning client in the federated learning setup, may be automatically overloaded on the next few lines...
         'dropout':0.1, #used in the model definition 0.0 mean, no unit is dropped out (all data is kept)
         'dataConfig':1, #set the data samples configuration to use, 1, 2 or 3
         'model':'cnn',#set the model to use, 'cnn' to use a convnet model or 'mlp' to rely on a sompler multilayer perceptron
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
if 'federated' in hparams.keys():
    if len(hparams['federated'])>0:
        enable_federated_learning=True

#activate XLA graph optimisation, if True, GPU AND CPU XLA is applied
useXLA=False

#the metric monitored to enable early stopping and learning rate decay
monitored_loss_name='val_sparse_categorical_crossentropy'

#profile some training steps to check pipeline processing time bottlenecks (from Tensorboard)
use_profiling=True

# define here the used model under variable 'model'
model_file='examples/federated/model_mnist_classif.py'

# activate weight moving averaging over itarations (Polyak-Ruppert)
weights_moving_averages=True

# random seed used to init weights, etc. Use an integer value to make experiments reproducible
random_seed=42

# WARNING, maybe adjust the number of epochs here
#-> in centralised learning, the number of epochs is the total number of epochs for a single training session
#-> in federated learning, the number of epochs is the total number of epochs for a single training round
nbEpoch=hparams['nbEpoch']

# stop condition, taking into account if val_loss does not decrease for early_stopping_patience epoch
early_stopping_patience=10

#set here paths to your data used for train, val -> only for config1
# -> the hyperparameter hparams['procID'] (i.e. federated client ID) is used to select the right data
maybe_download_data()
#switch from federated learning to centralised depending on hyperparameter setting
if enable_federated_learning:
    client_id = int(hparams['procID']) + 1 #client ID is 0-based while dataset is 1-based
    #look for a single file related to a specific client
    raw_data_dir_train = os.path.join(os.path.expanduser("~"),'.keras/datasets/mnist-data/', 'config'+str(hparams['dataConfig'])+'/client' + str(client_id) + '/data.csv')
else:
    #look for all client files as a single dataset as for centralised learning
    raw_data_dir_train = os.path.join(os.path.expanduser("~"),'.keras/datasets/mnist-data/', 'config'+str(hparams['dataConfig']))
raw_data_dir_val = os.path.join(os.path.expanduser("~"),'.keras/datasets/mnist-data/mnist_test.csv')

raw_data_filename_extension=''
nb_train_samples=6000 #manually adjust here the number of temporal items out of the temporal block size
nb_val_samples=10000
#if relying on centralized learning, the total amount of data is the sum of all client data
if not(enable_federated_learning):
    nb_train_samples*=10
    nb_val_samples
batch_size=32
steps_per_epoch=nb_train_samples//batch_size
validation_steps=nb_val_samples//batch_size
reference_labels=['category']

########## FEDERATED SERVER PARAMETERS SECTION ################
# -> by default federated server runs on a local machine with the clients and one generally like to experiment in simulation mode
# BUT if you need to move to real distributed federated learning with server and client on different machines, do this:
# 1. adjust the parameter below that specifies where the central server is: ip:port
# 2. start the federated on the target server machine WITHOUT the simulation option using the deeplearningtools.start_federated_server for instance:
# apptainer run /path/to/containers/tf2_addons.sif -m deeplearningtools.start_federated_server -u examples/federated/mysettings_mnist_classification.py -rounds 10
# 3. start each client with a given identifier (procID, here procID=1 for example) on their own devices using deeplearningtools.experiments_manager using command:
# apptainer run /path/to/containers/tf2_addons.sif -m deeplearningtools.start_federated_server -u examples/federated/mysettings_mnist_classification.py --procID 1
federated_learning_server_address="localhost:8080"

########## MODEL SERVING/PRODUCTION PARAMETERS SECTION ################
#-> server and port number and other parameters to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=5#timeout limit when a client requests a served model
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

  return {}

def get_learningRate():
  ''' define here the learning rate
  Returns a sclalar (float) or a scheduler
  '''
  if 'isFLserver' in hparams.keys():
    if hparams['isFLserver']==True:
      # no fine tuning on the centralised model
      return 0.0
  
  return hparams['learningRate']


def get_optimizer(model, loss, learning_rate):
    '''define here the specific optimizer to be used
    Returns a tensorflow optimizer object or a string defined in Tensorflow that targets a specific optimizer with default config
    '''

    optimizer=tf.keras.optimizers.Adam(hparams['learningRate'])

    return optimizer

def get_metrics(model, loss):
    return [tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False), metrics.ConfusionMatrix(num_classes=10)]

def get_total_loss(model):#inputs, model_outputs_dict, labels, weights_loss):
    '''a specific loss can be defined here or simply use a string that refers to a keras loss
    Args:
        model: the model to be optimized that may be used to focus loss on a set of specific layers or so
    Returns:
        a keras implemented loss represented by a string or a custom loss
        => it is recommended to return a tensor named 'loss' in order to enable some
        useful default options such as early stopping
    '''
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    return loss

def read_data(path, dropna=True):
    dataframe = pd.read_csv(path)
    if dropna:
        dataframe.dropna(inplace=True)
    return dataframe

def load_dataframes(path, dropna=True) -> list:
    if path.endswith('.csv') and os.path.isfile(path):
        return read_data(path, dropna=dropna)
    else:
        print('Looking for csv files in target folder ', path)
        files=extractFilenames(path, '*.csv')
        print('Agagating multiple csv files: ', files)
        dataframes=[read_data(file, dropna=dropna) for file in files]
        #concatenate dataframes
        return pd.concat(dataframes)

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
    
    dataframe = load_dataframes(raw_data_files_folder)
    features = dataframe.iloc[:,1:]
    targets = dataframe.iloc[:,:1]

    #convert each sample from shape (784,1) to (28,28,1) to allow for convolutional layers processing
    #also, data samples are normalised to range [0;1] (casted to float) instead of the original uint8 format with range [0;255]
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    def reshape_to_2D_normalise(sample, label):
        return tf.cast(tf.reshape(sample, [28, 28, 1]), tf.float32)/255.0, label
    dataset=dataset.map(reshape_to_2D_normalise)

    return dataset.shuffle(batch_size*20).batch(batch_size, drop_remainder=True).prefetch(1)

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

        @tf.function(input_signature=[tf.TensorSpec(shape=[1,28, 28, 1], dtype=tf.uint8, name=served_input_names[0])])
        def served_model(self, input_frame):
            ''' a decorated function that specifies the input data format, processing and output dict
            Args: input tensor(s)
            Returns a dictionnary of {'output key':tensor}
            '''
            pred=model(tf.cast(input_frame, tf.float32)/255.0)
            return {served_head_names[0]:pred}
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
        print('Trying to open a video stream...')
        self.video_capture=cv2.VideoCapture(0)#TODO, maybe replace value 0 by another camera index or a string targetting a video file
        #check if stream is correctely loaded
        assert self.video_capture.isOpened(), "Error opening video stream or file"
        print('Video stream opened')
        self.read_frame()

    def read_frame(self):
      ''' Reads a frame from the video stream
          Returns the read frame
          Raises ValueError if no frame available
      '''

      frame_is_ok, self.frame = self.video_capture.read()
      if not(frame_is_ok):
        raise ValueError('No input image available')
      return self.frame

    def getInputData(self, idx):
        ''' method that returns data samples complying with the placeholder
        defined in function get_input_pipeline_serving
        Args:
           idx: the input data index
        Returns:
           the data sample with shape and type complying with the server input
        '''
        #here, capture a frame from the webcam and apply a specific transform to comply with model expected input 
        #->see get_served_module function above: input should be of dimemsions (1,28,28,1)
        
        frame = cv2.resize(self.read_frame(), (28, 28))
        #convert to grayscale and add batch and color channel dimensions
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, (0,-1))
        return {served_input_names[0]:frame}

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
        self.ax1.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        #self.ax1.title('Input image')
        objects = tuple([i for i in range(10)])
        y_pos = np.arange(len(objects))
        self.ax2.bar(y_pos, response, tick_label=objects)#, align='center', alpha=0.5)
        self.ax2.set_title('Class probabilities')
        plt.pause(0.1)

    def finalize(self):
        ''' a function called when the prediction loop ends '''
        #print('Prediction process ended successfuly')
