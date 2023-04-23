'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs

Application : time series forecasting
FULL PROCESS USE EXAMPLE:
1. TRAIN/VAL : start a train/val session using command (a singularity container with an optimized version of Tensorflow is used here /path/to/tf2_addons.sif):
apptainer run --nv /path/to/tf2_addons.sif -m deeplearningtools.experiments_manager --usersettings=examples/timeseries/mysettings_timeseries_forecasting.py
---> this will create an experiment folder with all model checkpoints and training parameters history, say: /path/to/experiments/examples/timeseries/TS_trials_depth3_nstacks1_nlayers2_nneurons32_smoothedParamsTrue_nbEpoch2_bottleneckSize2_learningRate0.0005_tsLengthIn128_tsLengthOut10_nbChannels11_batchSize10_2022-12-05--18:01:39

2. SERVE MODEL : start a tensorflow model server on the produced eperiment models using command (the -psi command permits to start tensorflow model server installed in a singularity/apptainer container, /absolute/path/to/tf_server.sif):
python3 -m deeplearningtools.start_model_serving --model_dir=/absolute/path/to/experiments/examples/timeseries/TS_trials_depth3_nstacks1_nlayers2_nneurons32_smoothedParamsTrue_nbEpoch100_bottleneckSize2_learningRate0.0005_tsLengthIn128_tsLengthOut10_nbChannels11_batchSize10_2023-04-04--07:46:31/ -psi=/absolute/path/to/tf_server.sif

3. REQUEST MODEL : start a client that sends continuous requests to the server
apptainer run --nv /path/to/tf2_addons.sif -m deeplearningtools.experiments_manager --predict_stream=-1 --model_dir=/absolute/path/to/experiments/examples/timeseries/TS_trials_depth3_nstacks1_nlayers2_nneurons32_smoothedParamsTrue_nbEpoch100_bottleneckSize2_learningRate0.0005_tsLengthIn128_tsLengthOut10_nbChannels11_batchSize10_2023-04-04--07:46:31/
'''

#my imports
from deeplearningtools.helpers import loss as helpers_loss
from deeplearningtools.helpers import model as helpers_model

#libs imports
import tensorflow as tf
import numpy as np
import os

#-> set here your own working folder
workingFolder='experiments/examples/timeseries'

#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=20#timeout limit when a client requests a served model
#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='TS_trials'

hparams={'depth':3,
         'nstacks':1,
         'nlayers':2,
         'nneurons':32,
         'smoothedParams':True,
         'nbEpoch':100,
         'bottleneckSize':2,#the size of the bottleneck output size for each of the encoders
         'learningRate':0.0005, #TWEAKABLE
         'tsLengthIn':128,   #TWEAKABLE
         'tsLengthOut':10,   #TWEAKABLE
         'nbChannels': 11,
         'batchSize':10
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
#set here XLA optimisation flags, either tf.OptimizerOptions.OFF#ON_1#OFF
#XLA_FLAG=tf.OptimizerOptions.OFF#ON_1#OFF
useXLA=True

#profile some training steps to check pipeline processing time bottlenecks (from Tensorboard)
use_profiling=True

# define here the used model under variable 'model'
model_file='examples/timeseries/model_ts_TCN.py'
# CAN SWITCH TO/COMPARE WITH : model_file='examples/timeseries/model_ts_AE.py'

# activate weight moving averaging over itarations (Polyak-Ruppert)
weights_moving_averages=hparams['smoothedParams']

# random seed used to init weights, etc. Use an integer value to make experiments reproducible
random_seed=42

# learning rate decaying parameters
nbEpoch=hparams['nbEpoch']

# stop condition if val_loss does not decrease for early_stopping_patience epoch
early_stopping_patience=10

#set here paths to your data used for train, val
raw_data_dir_train = "../../../../datasamples/timeseries/"
raw_data_dir_val =   "../../../../datasamples/timeseries/" # WARNING, IN THIS DEMO TRAIN AND VAL DATA ARE THE SAME, DATASET MUST BE DISTINCT FOR REAL EXPERIMENTS
raw_data_filename_extension='*.csv'
temporal_series_length=hparams['tsLengthIn']+hparams['tsLengthOut']
ts_windowing_shift_ratio=10
ts_windowing_shift=temporal_series_length//ts_windowing_shift_ratio
nb_train_samples=(ts_windowing_shift_ratio*2000)//temporal_series_length#the 2 sample CSV files represent 2000 points
nb_val_samples=(ts_windowing_shift_ratio*2000)//temporal_series_length#the 2 sample CSV files represent 2000 points
batch_size=hparams['batchSize']
steps_per_epoch=nb_train_samples//batch_size
validation_steps=nb_val_samples//batch_size
reference_labels=['startDate', 'stopDate'] #to be used if many labels are generated by the get_input_pipeline_train_val function

########## LOCAL PARAMETERS (ONLY USED BELOW) SECTION ################
window_offset = 1
csv_field_delim=','
record_defaults=[['timestamp'], ['timestamp'], [0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
reference_labels=['startDate', 'stopDate', 'prevDayFreeDay', 'todayFreeDay', 'nextDayFreeDay'] #to be used if many labels are generated by the get_input_pipeline_train_val function
#record_defaults=[['timestamp'], ['timestamp'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ]
input_features_nb=len(record_defaults)-len(reference_labels)
print('input_features_nb:',input_features_nb)
labels_cols_nb=2 #on this last datasets, the 2 first columns are used as timestamps
na_value_string='N/A'
field_of_view=0
########## MODEL SERVING/PRODUCTION PARAMETERS SECTION ################
#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=1#timeout limit when a client requests a served model
served_input_names=['temporal_features', 'yesterday_isfree', 'today_isfree', 'tomorrow_isfree']
served_head_names=['prediction']

########## LOCAL PARAMETERS (ONLY USED BELOW) SECTION ################

''' Normalization variable '''
mean_vals = [1043.7930703279815, 937.5039896396833, 587.9572398695195, 42.70257611833159, 46.427388557764374, 41.9650561862, 8.372859834289358, 63.985666757595325, 23.704879281756785, 22.60299797330886, 24.366915626761347]
std_vals = [753.5115459652183, 625.5025770874278, 271.7532042774016, 7.292516897584224, 6.326844458353698, 6.974501426484209, 2.9139284071542098, 18.680419651268164, 2.292511632975457, 2.0780787914960706, 1.516894462249191]

########## TRAIN/VAL PERSONNALIZED FUNCTIONS SECTION ################

# add here any additionnal callback to use along the train/val process
def addon_callbacks(model, train_samples, val_samples):
  ''' optionnal callbacks can be defined here
  Arg: the defined model
  Returns a list of tf.keras.callbacks or an empty list
  '''
  return []

def get_learningRate():
  ''' define here the learning rate
  Returns a sclalar (float) of a scheduler
  '''
  return hparams['learningRate']
  '''tf.keras.optimizers.schedules.ExponentialDecay(
                                        initial_learning_rate=hparams['learningRate'],
                                        decay_steps=40,
                                        decay_rate=0.1,
                                        staircase=True)
  '''

def get_optimizer(model, loss, learning_rate):
    '''define here the specific optimizer to be used
    Returns a tensorflow optimizer object
    '''
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return optimizer

def get_metrics(model, loss):
  return 'mean_absolute_error'
  '''{
          'dense_1': tf.keras.metrics.mean_squared_error,
  }'''

def get_total_loss(model):#inputs, model_outputs_dict, labels, weights_loss):
    '''a specific loss for data reconstruction when dealing with autoencoders
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply with the ones defined in the model_file
        labels: the reference data / ground truth if available
        weights_loss: the model weights loss that may be used for regularization
    '''
    print('output=',model.outputs)
    #reconstruction_loss='mean_squared_error'
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
      print('y_true=',y_true)
      print('y_pred=',y_pred)
      return tf.keras.losses.MSE(y_pred, y_true)

    # Return a function or a string indicating a standard loss to be used, ex: 'mean_squared_error'
    return loss

'''Define here the input pipelines :
-1. a common function for train and validation modes
-2. a specific one for the serving model_extra_update_ops
'''
def get_input_pipeline(raw_data_files_folder, isTraining, batch_size, nbEpoch):
    ''' define an input pipeline able to load temporal series from a set of
    CSV files and a batch size specified as inputs
    TODO, look at the doc here : https://www.tensorflow.org/programmers_guide/datasets
    @param batch_size : the expected size of a batch
    @param raw_data_files_folder : the folder where files are stored
    @param shuffle_batches : a boolean that activates batch shuffling
    '''
    from deeplearningtools import DataProvider_input_pipeline #only import here to reduce dependencies in serving mode
    #load all csv files to use for training
    raw_data_files=DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_files_folder, file_extension=raw_data_filename_extension)
    #sort files in numeric order wrt the last integer before file extension
    raw_data_files=sorted(raw_data_files, key=lambda e: int((e.split('.')[-2]).split('_')[-1]))
    print('Input files found (SORTED IN TIME)='+str(raw_data_files))
    """def per_sample_process_function(single_period_data_block_raw, timestamps):
        ''' let the raw data as is but reduce the timestamp to the first and last date
        '''
        stack = [timestamps[0,0], timestamps[-1,temporal_series_length-1]]#tf.expand_dims(timestamps[0], axis=0), tf.expand_dims(timestamps[-1], axis=0)]
        #print(stack)
        timestamps_start_stop=tf.stack(stack)
        return single_period_data_block_raw[], single_period_data_block_raw#(single_period_data_block_raw, timestamps_start_stop)
    """
    def per_sample_process_function(single_period_data_block_raw, timestamps):
        ''' this custom function is intended to post process each sample independantly.
        Here, it separates it does multiple stuff:
        1) it separates input data and expected outcome (labels)
        2) normalizes the data with respect to the precomputed feature means and deviations measured on the train dataset
        3) it transforms oversampled metadata into simple scalars 
        '''
        print('### per_sample_process_function START')
        print('single_period_data_block', single_period_data_block_raw)
        single_period_data_block_raw=tf.transpose(single_period_data_block_raw)
        timestamps=tf.transpose(timestamps)
        freedays=single_period_data_block_raw[:,-3:]
        print('Free days=',freedays)
        print('single_period_data_block_raw', single_period_data_block_raw[:,:-3])

        yesterday_isfree = freedays[0,0]
        today_isfree = freedays[0,1]
        tomorrow_isfree = freedays[0,2]

        #normalizing data wrt mean and std measured on the train set
        means=tf.constant(mean_vals, shape=[1,len(mean_vals)])
        vars=tf.constant(std_vals, shape=[1,len(mean_vals)])
        ts_all=tf.divide(tf.math.subtract(single_period_data_block_raw[:,:-3],means),vars+1e-6)
        print('ts_all', ts_all)
        print('timestamps',timestamps)
        #stacking labels
        stack = [timestamps[0,0],
                 timestamps[hparams['tsLengthIn']-1,-1],
                 tf.as_string(yesterday_isfree),
                 tf.as_string(today_isfree),
                 tf.as_string(tomorrow_isfree)]

        stack=(ts_all[:hparams['tsLengthIn'],:],
                yesterday_isfree,
                today_isfree,
                tomorrow_isfree
                ), ts_all[hparams['tsLengthIn']:,:] #try here to predict all features future #alternative, focus on a single feature : tf.slice(ts_all, begin=[hparams['tsLengthIn'], 0], size=[hparams['tsLengthOut'],1])
        print("sample content", stack)
        print('### per_sample_process_function END')
        return stack
         #(single_period_data_block_raw[:-3,:],timestamps_start_stop_freedays)
    dataset = DataProvider_input_pipeline.FileListProcessor_csv_time_series(files=raw_data_files,
                                                csv_field_delim=csv_field_delim,
                                                record_defaults_values=record_defaults,
                                                batch_size=batch_size,
                                                epochs=nbEpoch,
                                                temporal_series_length=temporal_series_length,#i.e.hparams['tsLengthIn']+hparams['tsLengthOut']
                                                windowing_shift=ts_windowing_shift,
                                                na_value_string=na_value_string,
                                                labels_cols_nb=labels_cols_nb,
                                                per_sample_preprocess_fn=per_sample_process_function,
                                                selected_cols=None)#, postprocess_fn=process_function)

    print('dataset', dataset)

    return dataset
'''
################################################################################
## Serving (production) section, define here :
-get_served_module():  define how the model will be applied on production data, applying custom pre and post processing
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
'''
time_series_input_serving_shape=[None, hparams['tsLengthIn'], input_features_nb]
def get_served_module(model, model_name):
  ''' following https://www.tensorflow.org/guide/saved_model
      Create a custom module to specify how the model will be used in production/serving
      specific preprocessing can be defined as well as post-processing
  '''
  class ExportedModule(tf.Module):
    def __init__(self, model):
      super().__init__()
      self.model=model

    @tf.function(input_signature=[tf.TensorSpec(shape=time_series_input_serving_shape, dtype=tf.float32, name=served_input_names[0]),#'temporal_features'),
                                  tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name=served_input_names[1]),#'yesterday_isfree'),
                                  tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name=served_input_names[2]),#'today_isfree'),
                                  tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name=served_input_names[3])])#'tomorrow_isfree')])
    def served_model(self, input, yesterday_isfree, today_isfree, tomorrow_isfree):
      ''' a decorated function that specifies the input data format, processing and output dict
        Args: input tensor(s)
        Returns a dictionnary of {'output key':tensor}
      '''
      pred =model([input, yesterday_isfree, today_isfree, tomorrow_isfree])
      print('Exporting model output :', pred)
      return {served_head_names[0]:pred}
  return ExportedModule(model)

import glob #for filesystem exploration tools
import pandas as pd #for csv files easy reading and timeseries processing
import matplotlib.pyplot as plt

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
        self.debugMode=debugMode
        if self.debugMode is True:
            print('RPC Client ready to interract with the server')

        filenames=glob.glob(os.path.join(os.getcwd(),'../../../../../',raw_data_dir_val,'*.csv'))
        print('attempting to find csv file from : '+str(filenames))
        assert len(filenames)>0, 'No files found, check your settings'
        if debugMode is True:
          print('Files found='+str(len(filenames)))
          print('attempting to load file : '+str(filenames[0]))

        self.inputdata=pd.read_csv(filenames[0], delimiter=csv_field_delim).values[:,1:]
        if debugMode is True:
          print('--> loaded data : '+str(self.inputdata))
          print('--> loaded data shape: '+str(self.inputdata.shape))
        #print('Read text data, shape='+str(self.inputdata.shape))
        self.neighborhood_range=2
        self.current_time_idx=self.neighborhood_range*temporal_series_length+field_of_view
        #prepare plots
        self.firstCall=True
        self.fig, (self.ax1) = plt.subplots(1, 1, sharex=True, sharey=True) #add some more subplots if required
        
        #prepare normalization values fof the time series
        self.mean_vals=np.expand_dims(mean_vals,0)
        self.std_vals=np.expand_dims(std_vals,0)+1e-6

    def getInputData(self, idx):
        ''' method that returns data samples complying with the placeholder
        defined in function get_input_pipeline_serving
        Args:
           idx: the input data index
        Returns:
           the data sample with shape and type complying with the server input
        '''
        print('time index=', idx)
        #get the x curve ticks and related data block
        t_start=self.current_time_idx-hparams['tsLengthIn']*self.neighborhood_range
        t_stop=self.current_time_idx+hparams['tsLengthOut']*self.neighborhood_range
        self.current_data_block=self.inputdata[t_start:t_stop,:]
        self.x=self.current_data_block[:,0]
        self.target=self.current_data_block[:,1:]
        '''one have here a temporal block from (t-tsLenghtIn*neighborhood_range to (t+tsLengthOutneighborhood_range)
        -> i.e. a temporal block of size tsLenghtIn*neighborhood_range+tsLenghtOut*neighborhood_range
        '''
        #print('self.target.shape', self.target.shape)
        
        #focus on the request
        present_start=hparams['tsLengthIn']*self.neighborhood_range #an offset from tt_start to reach the index of self.current_time_idx
        present_stop=present_start+temporal_series_length
        request_t_start=present_start-hparams['tsLengthIn']
        request_t_stop=present_start
        
        self.x_current=self.x[present_start:present_stop]
        #print('x.current='+str(self.x_current))
        #increment time steps
        self.current_time_idx+=1

        if self.debugMode is True:
            print('Generating input features (random values) of shape '+str(self.target.shape))
            print('selecting temporal window between indexes:',(request_t_start,request_t_stop))
            print('request=',self.target[request_t_start:request_t_stop,:].shape)

        temporal_features_request=self.target[request_t_start:request_t_stop,:11]
        temporal_features_expected_answer=self.target[request_t_stop:request_t_stop+hparams['tsLengthOut'],:11]
        yesterday_isfree=self.target[request_t_start:request_t_stop,11]
        today_isfree=self.target[request_t_start:request_t_stop,12]
        tomorrow_isfree=self.target[request_t_start:request_t_stop,13]
        yesterday_isfree=np.array(yesterday_isfree[0]).reshape((1,1))
        today_isfree=np.array(today_isfree[0]).reshape((1,1))
        tomorrow_isfree=np.array(tomorrow_isfree[0]).reshape((1,1))

        #send data to estimation server
        self.request_xticks=self.x[request_t_start:request_t_stop]
        self.request_values=temporal_features_request#.reshape(time_series_input_serving_shape[1:])
        
        #prepare expected answer for demo/analysis
        self.answer_xticks=self.x[request_t_stop:request_t_stop+hparams['tsLengthOut']]
        self.expected_answer_values=temporal_features_expected_answer
        #normalize temporal series wrt mean and std values for each modality
        self.request_values=(self.request_values-self.mean_vals)/self.std_vals
        self.expected_answer_values=(self.expected_answer_values-self.mean_vals)/self.std_vals
        #add batch dimension
        self.request_values=np.expand_dims(self.request_values,0)
        return {served_input_names[0]:self.request_values.astype(np.float32),
                served_input_names[1]:yesterday_isfree.astype(np.float32),
                served_input_names[2]:today_isfree.astype(np.float32),
                served_input_names[3]:tomorrow_isfree.astype(np.float32),

        }
        

    def decodeResponse(self, result):
        ''' receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        '''
        response_recons = np.reshape(np.array(result.outputs[served_head_names[0]].float_val),
                              [1, hparams['tsLengthOut'], 11]).astype(np.float32)
        #print('Query shape='+str(self.request_values.shape))
        #print('Answer shape='+str(response_recons.shape))
        
        def standardize_timeseries(features):
          #per channel standardization as done at the entry of the model
          return (features - np.mean(features, axis=0))/(np.std(features, axis=0)+1e-6)

        self.ax1.cla()
        #print('self.request_xticks', self.request_xticks.shape)
        #print('self.answer_xticks', self.answer_xticks.shape)
        timeseries_ticks=np.concatenate([self.request_xticks,self.answer_xticks], axis=0)
        #print('timeseries_ticks', timeseries_ticks.shape)
        timeseries_values=np.concatenate([self.request_values[0],self.expected_answer_values], axis=0)
        #print('timeseries_values', timeseries_values.shape)
        self.ax1.plot(timeseries_ticks, timeseries_values)#standardize_timeseries(self.request_values[0,:])))
        #plot request time series
        self.ax1.set_title('input signals (per channel standardized)')
        
        #plot predicted future
        self.ax1.plot(self.answer_xticks, response_recons[0])
        plt.pause(0.02)

    def finalize(self):
        ''' a function called when the prediction loop ends '''
        print('Prediction process ended successfuly')
