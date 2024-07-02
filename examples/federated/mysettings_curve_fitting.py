r"""
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
==> application : noisy curve regression

FULL PROCESS USE EXAMPLE:
1. TRAIN/VAL : start a train/val session using command (a singularity container with an optimized version of Tensorflow is used here):
-> start model parameter server :
apptainer run tf2_addons.sif -m deeplearningtools.start_federated_server --usersettings=examples/federated/mysettings_curve_fitting.py

-> start multiple learning clients
apptainer run --nv tf2_addons.sif -m deeplearningtools.experiments_manager --usersettings=examples/federated/mysettings_curve_fitting.py --procID=1
apptainer run --nv tf2_addons.sif -m deeplearningtools.experiments_manager --usersettings=examples/federated/mysettings_curve_fitting.py --procID=-1

OR simply start a federated session in simulation mode (all processed (parameter server and clients shared on the same machine)
apptainer run tf2_addons.sif -m deeplearningtools.start_federated_server -sim --usersettings=examples/federated/mysettings_curve_fitting.py

2. SERVE MODEL : start a tensorflow model server on the produced eperiment models using command (the -psi command permits to start tensorflow model server installed in a singularity container):
python3 -m deeplearningtools.start_model_serving --model_dir /home/alben/workspace/listic-deeptool/experiments/examples/federated/my_test_hiddenNeurons50_predictSmoothParamsTrue_learningRate0.001_nbEpoch5000_addNoiseTrue_anomalyAtX-4_procID-1_2021-06-13--16\:43\:22/ -psi /home/alben/install/containers/tf_server.cpu.sif

3. REQUEST MODEL : start a client that sends continuous requests to the server
apptainer run --nv tf2_addons.sif -m deeplearningtools.experiments_manager --predict_stream=-1 --model_dir /home/alben/workspace/listic-deeptool/experiments/examples/federated/my_test_hiddenNeurons50_predictSmoothParamsTrue_learningRate0.001_nbEpoch5000_addNoiseTrue_anomalyAtX-4_procID-1_2021-06-13--16\:43\:22/

Check training logs : apptainer exec --nv /path/to/tf2_addons.sif tensorboard --logdir experiments/examples/federated/regression
if running in simulation mode, logs are placed in /tmp/ray/session_latest/runtime_resources/working_dir_files/_ray_pkg_blahblah/logs
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#-> set here your own working folder
workingFolder='experiments/examples/federated/regression'

#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='my_test'

""" define here some hyperparameters to adjust the experiment
===> Note that this dictionnary will complete the session name
"""
hparams={
         'federated':'FedAvg',#set '' if not making use of federated learning or set the flower strategy name of a custom one from deeplearningtools.helpers
         'minCl':30,#minimum number of clients to allow for federated learning
         'minFit':3,#minimum number of clients to allow for a federated learning fitting round
         'hiddenNeurons':10,#set the number of neurons per hidden layers
         'predictSmoothParams':True, #set True to activate parameters moving averages use for prediction
         'learningRate':0.01,
         'nbEpoch':1, #when in federated learning mode, this represents the client number of local epochs to be performed for each round
         'addNoise':True, #set True to add noise to the target curve
         'range':0.5,
         'procID':0, #index of learning client in the federated learning setup, may be automatically overloaded on the next few lines...
         'clusteringMethod':'SpectralClustering' #choose among available options, currently SpectralClustering, MADC, EDC
        }

"""'set the list of GPUs involved in the process. HOWTO:
->if using CPU only mode, let an empty list
->if using a single GPU, only the first ID of the list will be considered
->if using multiple GPUs, each GPU ID will be considered
=> general recommendation: always try to focus on unused GPUs to avoid conflicts
with other processing jobs, yours and the ones of your colleagues.
Then, connect to the processing node and type in command line 'nvidia-smi'
to check which gpu is free (very few used memory and GPU )
"""
used_gpu_IDs=[]
#activate XLA graph optimisation, if True, GPU AND CPU XLA is applied
useXLA=True

#profile some training steps to check pipeline processing time bottlenecks (from Tensorboard)
use_profiling=True

#activate federated learning if required
enable_federated_learning=False
if len(hparams['federated'])>0:
  enable_federated_learning=True

# define here the used model under variable 'model'
model_file='examples/federated/model_curve_fitting.py'

# activate weight moving averaging over itarations (Polyak-Ruppert)
weights_moving_averages=True

# random seed used to init weights, etc. Use an integer value to make experiments reproducible
random_seed=41

# stop condition, taking into account if val_loss does not decrease for early_stopping_patience epoch
nbEpoch=hparams['nbEpoch']
early_stopping_patience=10
monitored_loss_name='val_mean_squared_error'
#set here paths to your data used for train, val
raw_data_dir_train = ''
raw_data_dir_val = ''
raw_data_filename_extension=''
nb_train_samples=1000 #manually adjust here the number of temporal items out of the temporal block size
nb_val_samples=1000
steps_per_epoch=100
validation_steps=100
batch_size=10
reference_labels=['values']

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
#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='localhost'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=1#timeout limit when a client requests a served model
serve_on_gpu=True #uncomment to activate model serving on GPU instead of CPU
served_input_names=['input']
served_head_names=['prediction']

########## LOCAL PARAMETERS (ONLY USED BELOW) SECTION ######
served_head_names=['prediction']

########## LOCAL PARAMETERS (ONLY USED BELOW) SECTION ################
def target_curve(x):
    #y=(x/2-1)**3+3*(x/2-1)**2-2
    y = (np.sin(x) + x)/hparams['minCl']

    if hparams['addNoise'] is True:
        noise=np.random.normal(loc=0.0, scale=1/(2*hparams['minCl']), size=x.shape).astype(np.float32)
        y+=noise
    return y

########## TRAIN/VAL PERSONNALIZED FUNCTIONS SECTION ################
# add here any additionnal callback to use along the train/val process
def addon_callbacks(model, train_samples, val_samples):
  """ optionnal callbacks can be defined here
  Arg: the defined model
  Returns a list of tf.keras.callbacks or an empty list
  """
  return {}

def get_learningRate():
  """ define here the learning rate
  Returns a sclalar (float) or a scheduler
  """
  if 'isFLserver' in hparams.keys():
    if hparams['isFLserver']==True:
      # no fine tuning on the centralised model
      return 0.0
  return hparams['learningRate']

def get_optimizer(model, loss, learning_rate):
    """define here the specific optimizer to be used
    Returns a tensorflow optimizer object
    """
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    return optimizer

def get_metrics(model, loss):
  return {
          'dense_1': [tf.keras.metrics.MAE, tf.keras.metrics.mean_squared_error],
          }

def get_total_loss(model):#inputs, model_outputs_dict, labels, weights_loss):
    """a specific loss can be defined here or simply use a string that refers to a keras loss
    Args:
        model: the model to be optimized that may be used to focus loss on a set of specific layers or so
    Returns:
        a keras implemented loss represented by a string or a custom loss
        => it is recommended to return a tensor named 'loss' in order to enable some
        useful default options such as early stopping
    """

    reconstruction_loss=tf.keras.losses.MSE
    return reconstruction_loss

"""Define here the input pipelines :
-1. a common function for train and validation modes
-2. a specific one for the serving model_extra_update_ops
"""
def get_input_pipeline(raw_data_files_folder, isTraining, batch_size, nbEpoch):
    """ define an input pipeline a basic example here, define random x values
    associated to y=f(x) values to regress
    TODO, look at the doc here : https://www.tensorflow.org/programmers_guide/datasets
    @param raw_data_files_folder : the variable that could target a dataset/folder...
    @param isTraining : a boolean that activates batch shuffling
    """
    """ nb clients = hparams['minCl'] ; current client in range [0, hparams['minCl']-1]]
    -> sampling interval = [procID-minCl/2-range, procID-minCl/2+range] """
    sampling_center=int(int(hparams['procID'])-int(hparams['minCl'])/2)
    sampling_interval_min=sampling_center-int(hparams['range'])
    sampling_interval_max=sampling_center+int(hparams['range'])

    dataset_sample=nb_train_samples if isTraining else nb_val_samples
    x=np.random.uniform(size=dataset_sample, low=sampling_interval_min, high=sampling_interval_max)
    targets = target_curve(x)
    dataset = tf.data.Dataset.from_tensor_slices((x, targets)).batch(batch_size).prefetch(1)
    return dataset

"""
################################################################################
## Serving (production) section, define here :
-get_served_module():  define how the model will be applied on production data, applying custom pre and post processing
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
"""
def get_served_module(model, model_name):
  """ following https://www.tensorflow.org/guide/saved_model
      Create a custom module to specify how the model will be used in production/serving
      specific preprocessing can be defined as well as post-processing
  """
  class ExportedModule(tf.Module):
    def __init__(self, model):
      super().__init__()
      self.model=model

    @tf.function(input_signature=[tf.TensorSpec(shape=[batch_size, 1], dtype=tf.float32, name=served_input_names[0])])
    def served_model(self, input):
      """ a decorated function that specifies the input data format, processing and output dict
        Args: input tensor(s)
        Returns a dictionnary of {'output key':tensor}
      """
      pred=model(input)
      return {served_head_names[0]:pred}
  return ExportedModule(model)

class Client_IO:
    """ A specific class dedicated to clients that need to interract with
    a Tensorflow server that runs the above model
    --> must have the following methods:
    def __init__(self, clientInitSpecs, debugMode): constructor that receives a debug flag
    def getInputData(self, idx): that generates data to send to the server
    def decodeResponse(self, result): that receives the response
    def finalize(self): the method call at the end of the process
    """
    def __init__(self, clientInitSpecs={}, debugMode=False):
        """ constructor
            Args:
               clientInitSpecs: a dictionnary to setup the client is necessary
               debugMode: set True if some debug messages should be displayed
        """

        self.debugMode=debugMode
        if self.debugMode is True:
            print('RPC Client ready to interract with the server')

        self.fig, self.ax = plt.subplots()

    def getInputData(self, idx):
        """ method that returns data samples complying with the placeholder
        defined in function get_input_pipeline_serving
        Args:
           idx: the input data index
        Returns:
           the data sample with shape and type complying with the server input
        """
        #here, only random numbers
        self.x=np.reshape(np.linspace(start=-1*float(hparams['minCl'])/5, stop=float(hparams['minCl'])/5, num=100).astype(np.float32), [-1,1])
        self.target=target_curve(self.x)
        if self.debugMode is True:
            print('Generating input features (random values) of shape '+str(self.target.shape))
        return {served_input_names[0]:self.x}

    def decodeResponse(self, result):
        """ receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        """
        response = np.array(result.outputs[served_head_names[0]].float_val)
        if self.debugMode is True:
            print('server model output=',result.outputs)
            print('request shape='+str(self.x.shape))
            print('Answer shape='+str(response.shape))
        self.ax.cla()
        self.ax.plot(self.x, self.target,'r+')
        self.ax.plot(self.x, response, 'b+')
        plt.pause(1)

    def finalize(self):
        """ a function called when the prediction loop ends """
        print('Prediction process ended successfuly')
