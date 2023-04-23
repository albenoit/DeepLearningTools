"""
#What's that ?
A set of script that demonstrate the use of Tensorflow experiments and estimators on different data types for various tasks
@brief : the main script that enables training, validation and serving Tensorflow based models merging all needs in a
single script to train, evaluate, export and serve.
taking large inspirations of official tensorflow demos.
@author : Alexandre Benoit, LISTIC lab, FRANCE

Several ideas are put together:
-estimators to manage training, valiation and export in a easier way
-using moving averages to store parameters with values smoothed along the last training steps (FIXME : ensure those values are used for real by the estimator, actually the graph shows 2 parameter savers...).
-visualization including embeddings projections to observe some data projections on the TensorBoard
-tensorflow-serving api use to serve the model and dynamically load updated models
-some tensorflow-serving client codes to reuse the trained model on single or streaming data

#How tu use it ?

The main script is experiments_manager.py can be used in 3 modes, here are some command examples:
1. train a model in a context specified in a parameters script such as mysettings_1D_experiments.py:
-> python experiments_manager.py --usersettings=mysettings_1D_experiments.py
2. start a tensorflow server on the trained/training model :
2.a if tensorflow_model_server is installed on the system
-> python experiments_manager.py --start_server --model_dir=experiments/1Dsignals_clustering/my_test_2018-01-03--14:40:53
2.b  if tensorflow_model_server is installed on a singularity container
-> python experiments_manager.py --start_server --model_dir=experiments/1Dsignals_clustering/my_test_2018-01-03--14:40:53 -psi=/patg/to/tf_server.sif
3. interract with the tensorflow server, sending input buffers and receiving answers
-> python experiments_manager.py --predict --model_dir=experiments/1Dsignals_clustering/my_test_2018-01-03--14\:40\:53/

NOTE : once trained (or along training), start the Tensorbdownscaledoard to parse logs of
the experiments folder (provided example is experiments/1Dsignals_clustering):
from the scripts directory using command: tensorboard  --logdir=experiments/1Dsignals_clustering
Then, open a web brwser and reach http://127.0.0.1:6006/ to monitor training
values and observe the obtained embeddings

#DESIGN:

1. The main code for training, validation and prediction is specified in the main script (experiments_manager.py).
2. Most of the use case specific parameters and Input/Output functions have been
moved to a separated settings script such as 'mysettings_1D_experiments.py' that
is targeted when starting the script (this filename is set in var FLAGS.usersettings in the main script).
3. The model to be trained and served is specified in a different script targetted in the settings file.

#KNOWN ISSUES :

This script has some known problems, any suggestion is welcome:
-moving average parameters reloading for model serving is not optimized, this should be enhanced.
-for now tensorflow_server only works on CPU so using GPU only for training and validation. Track : https://github.com/tensorflow/serving/issues/668

#TODO :

To adapt to new use case, just duplicate the closest mysettingsxxx file presented in the examples folder and adjust the configuration.
For any experiment, the availability of all the required fields in the settings file is checked by the tools/experiments_settings.py script.
You can have a look there to ensure you prepared everything right, some variables and functions must exist while some others are optionnal.

As a reminder, here are the functions prototypes:

-define a model to be trained and served in a specific file and follow this prototype:
--report model name in the settings file using variable name model_file or thecify a premade estimator using variable name premade_estimator
--def model( usersettings) #receives the external parameters that may be used to setup the model (number of classes and so depending on the task)
            mode), #mode set to switch between train, validate and inference mode
            wrt tf.estimator.tf.estimator.ModeKeys values
          => the returns a tf.keras.Model
          NOTE : custom models with specific loss can be used, tutorial here  https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
          
-def data_preprocess(features, model_placement)
-def postprocessing_before_export_code(code)
-def postprocessing_before_export_predictions(predictions)
-def getOptimizer(model, loss, learning_rate, global_step)
-def get_total_loss(inputs, predictions, labels, embedding_code, weights_loss)
-def get_validation_summaries(inputs, predictions, labels, embedding_code)
-def get_eval_metric_ops(inputs, predictions, labels, embedding_code)
-def get_input_pipeline_train_val(batch_size, raw_data_files_folder, shuffle_batches)
-def get_input_pipeline_serving()
-define the Client_IO class that presents at least those three methods:
---def __init__(self, debugMode):
---def getInputData(self, idx):
---def decodeResponse(self, result):
---def finalize():
-------> Note, the finalize method will be called once the number of expected
iterations is reached and if any StopIteration exception is sent by the client
-OPTIONNAL: add the dictionnary named 'hparams' in this settings file to carry those specific hyperparameters to the model
and to complete the session name folder to facilitate experiments tracking and comparison

Some examples of such functions are put in the README.md and in the versionned examples folder
"""

#script imports
from tools.experiment_settings import ExperimentSettings
import tools.experiments_settings_surgery
from tools.command_line_parser import get_commands
from tools.experiment_settings import define_callbacks
import os, shutil
import datetime, time
import tensorflow as tf
import numpy as np
import pandas as pd
import importlib
import types
import copy
import configparser
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python import debug as tf_debug
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from tensorflow.python.tools import optimize_for_inference_lib
from helpers import model_serving_tools
from helpers.model import track_gradients, track_weights_change
from helpers import federated
import helpers.tensor_msg_io
import helpers.kafka_io

import tools.gpu
try:
  import tensorflow_addons as tfa
except:
  print('WARNING, tensorflow_addons could not be loaded, this may generate errors for model optimization but should not impact model serving')

try:
  import tensorflow_model_optimization as tfmot
except:
  print('WARNING, tensorflow_model_optimization could not be loaded, this may generate errors for model optimization for instance if usersettings.tensorflow_model_optimization=True')


federated_learning_available=False
try:
  import flwr as fl
  federated_learning_available=True
except ModuleNotFoundError as e :
  print('WARNING, Flower lib no present, this may impact distributed learning if required. Error=',e)

from tensorflow.keras import mixed_precision

#constants
SETTINGSFILE_COPY_NAME='experiment_settings.py'
WEIGHTS_MOVING_AVERAGE_DECAY=0.998

def loadModel_def_file(usersettings, absolute_path=False):
  ''' basic method to load the model targeted by usersettings.model_file
  Args: sessionFolder, the path to the model file
  Returns: a keras model
  '''
  if absolute_path == False:
    model_path=os.path.basename(usersettings.model_file)
  if absolute_path == True:
    model_path=usersettings.model_file
  try:
    spec=importlib.util.spec_from_file_location('model_def', model_path)
    model_def = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_def)
  except Exception as e:
    raise ValueError('loadModel_def_file: Failed to load model file {model}, error message={err}'.format(model=usersettings.model_file, err=e))
  model=model_def.model

  print('loaded model file {file}'.format(file=model_path))
  return model


def loadExperimentsSettings(filename, call_from_session_folder=False, restart_from_sessionFolder=None, isServingModel=False):
    ''' load experiments parameters from the mysettingsxxx.py script
        also mask GPUs to only use the ones specified in the settings file
      Args:
        filename: the settings file, if restarting an interrupted training session, you should target the experiments_settings.py copy available in the experiment folder to restart"
        restart_from_sessionFolder: [OPTIONNAL] set the  session folder of a previously interrupted training session to restart
        isServingModel: [OPTIONNAL] set True in the case of using model serving (server or client mode) so that some settings are not checked
    '''

    if restart_from_sessionFolder is not None:
      if os.path.exists(restart_from_sessionFolder):
        print('Attempting to restart a previously ran training job...')
        sessionFolder=restart_from_sessionFolder
        #target the initial experiments settings file
        filename=os.path.join(restart_from_sessionFolder, SETTINGSFILE_COPY_NAME)
        print('From working folder'+str(os.getcwd()))
        print('looking for '+str(filename))
        if os.path.exists(filename):
          print('Found')
        else:
          raise ValueError('Could not find experiment_settings.py file in the experiment folder:'+str(sessionFolder))
      else:
        raise ValueError('Could not restart interrupted training session, working folder not found:'+str(restart_from_sessionFolder))
    else:
      print('Process starts...')

    usersettings=ExperimentSettings(filename, isServingModel, call_from_session_folder)

    if isServingModel:
      sessionFolder=os.path.dirname(filename)

    #manage the working folder in the case of a new experiment
    workingFolder=usersettings.workingFolder
    if restart_from_sessionFolder is None:
      sessionFolder=os.path.join(workingFolder, usersettings.session_name+'_'+datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
      usersettings.recoverFromCheckpoint=False
    else:
      usersettings.recoverFromCheckpoint=True

    print('Considered usersettings.hparams=',str(usersettings.hparams))
    return usersettings, sessionFolder

# Define and run experiment ###############################
def build_run_training_session(cid: str=None):
  ''' define and run the optimisation process
      this function loads settings from the working directory
      and builds/run the optimisation.
      -> This process can be the full training procedure 
      OR this can be an ephemeral intermediate training step
      as for transfer learning or federated learning
  '''

  #####################################
  # load configuration, expects the process working directory
  # to contain all the necessary files including:
  # - the configuration file (experiment_settings.py)
  # - optionally a 'checkpoints' folder in order to pursue training (recover from previous interruption) 
  settings_file=os.path.join(os.getcwd(), SETTINGSFILE_COPY_NAME)
  if cid!=None:
    with open('/tmp/'+str(cid)+'.notes', 'a') as f:
      message='\ncid {cid} with cwd={cwd} and job_session_folder in locals={l} or job_session_folder in globals={g}'.format(cid=cid,
                                                                                                                          cwd=os.getcwd(),
                                                                                                                          l='job_session_folder' in locals(),
                                                                                                                          g='job_session_folder' in globals())
      f.write(message) 

    settings_file=tools.experiments_settings_surgery.insert_additionnal_hparams(settings_file, {'procID':cid})

  #load experiment settings from current working directory
  usersettings, _ =loadExperimentsSettings(filename=settings_file,
                                           call_from_session_folder=True)
  gpu_workers_nb=tools.gpu.check_GPU_available(usersettings)

  if os.path.exists(os.path.join(os.getcwd(),'checkpoints')):
    print('Recovering training from checkpoint...')
    usersettings.recoverFromCheckpoint=True
  else:
    usersettings.recoverFromCheckpoint=False
  
  #####################################
  # define the input pipepelines (train/val)
  with tf.name_scope('Input_pipeline'):
    train_data =usersettings.get_input_pipeline(raw_data_files_folder=usersettings.raw_data_dir_train,
                                                      isTraining=True, batch_size=usersettings.batch_size,
                                                      nbEpoch=usersettings.nbEpoch)
    val_data = usersettings.get_input_pipeline(raw_data_files_folder=usersettings.raw_data_dir_val,
                                                      isTraining=False, batch_size=usersettings.batch_size,
                                                      nbEpoch=usersettings.nbEpoch)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++train_data', train_data)
    #data_it=train_data.as_numpy_iterator()
    #print("ELEMENTS", list(data_it))
    # if reading from a kafka log queue:
    if usersettings.consume_data_from_kafka: 
      print("Original dataset specs:\n->", train_data.element_spec)
      train_val_dataset_features=helpers.tensor_msg_io.get_data_label_features_from_dataset(train_data)
      print("train_val_dataset_features:",train_val_dataset_features)
      log_queue_name=usersettings.session_name
      if 'procID' in usersettings.hparams.keys():
        log_queue_name+=str(usersettings.hparams['procID'])
      kafka_reader_train=helpers.kafka_io.KafkaIO(topic_name=log_queue_name+'train', bootstrap_servers=usersettings.kafka_bootstrap_servers, element_spec=train_data.element_spec)
      kafka_reader_val=helpers.kafka_io.KafkaIO(topic_name=log_queue_name+'val', bootstrap_servers=usersettings.kafka_bootstrap_servers, element_spec=val_data.element_spec)
      #setup consumer depending on data and label types:
      if isinstance(train_data.element_spec[0], dict):
        train_data=kafka_reader_train.kafka_dataset_consumer_tf_custom(train_val_dataset_features, batch_size=usersettings.batch_size, shuffle=True)
      else:
        train_data=kafka_reader_train.kafka_dataset_consumer_tf_basic(batch_size=usersettings.batch_size, shuffle=True)
      if isinstance(val_data.element_spec[1], dict):
        val_data=kafka_reader_val.kafka_dataset_consumer_tf_custom(train_val_dataset_features, batch_size=usersettings.batch_size, shuffle=False)
      else:
        val_data=kafka_reader_train.kafka_dataset_consumer_tf_basic(batch_size=usersettings.batch_size, shuffle=False)
      print('------------------------------------------------------------train_data', train_data)
      print("KAFKA dataset specs:\n->", train_data.element_spec)
      print('Kafka connectors ready !')
  try:
    print('Train dataset size=', train_data.cardinality().numpy())
    print('Validation dataset size=', val_data.cardinality().numpy())
    train_iterations_per_epoch=train_data.n//usersettings.batch_size
    val_iterations_per_epoch=val_data.n//usersettings.batch_size
  except Exception as e:
    print('Could not estimate dataset sizes from input data pipeline, relying on settings nb_train_samples and nb_val_samples.')
    train_iterations_per_epoch=usersettings.nb_train_samples//usersettings.batch_size
    val_iterations_per_epoch=usersettings.nb_val_samples//usersettings.batch_size

      
  if usersettings.enable_mixed_precision:
    # use AMP
    print('Using Automatic Mixed Precision along the optimization process')
    print('### HINT : to make sure Tensor cores are used, and obtain faster processing, ensure that your kernels are multiples of 8 !')
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

  #####################################
  #create the model from the user defined model file
  # -> (script targeted by usersettings.model_file)
  
  #if current session folder contains the checkpoint
  initial_epoch=0
  if usersettings.recoverFromCheckpoint is False:

    print('**** Training from scratch...')
    model_scope=tf.name_scope('model')
    if gpu_workers_nb>1:
      print('Deploying model in a multi/distributed GPU training scheme')
      distribution_strategy=getattr( tf.distribute, usersettings.distribution_strategy)()
      print('-> Chosen distribution strategy :',distribution_strategy)
      model_scope=distribution_strategy.scope()#(model_scope, distribution_strategy.scope())
    usersettings.summary()


    #def load_fit_model():
    with model_scope:
      #load model
      model=loadModel_def_file(usersettings)(usersettings)
      #setup training
      learning_rate=usersettings.get_learningRate()
      loss=usersettings.get_total_loss(model)
      optimizer=usersettings.get_optimizer(model, loss, learning_rate)
      if usersettings.weights_moving_averages:
          print('Overriding optimizer option and enabling exponential weights moving average (EMA) along training...')
          ema_momentum_default=0.99
        
          if hasattr(optimizer, 'ema'):
            optimizer.use_ema=True
            optimizer.ema_momentum=ema_momentum_default
          else:
            try:
              print('Optimizer does not support EMA, trying to introduce EMA from tfa module')
              optimizer=tfa.optimizers.MovingAverage(optimizer, average_decay=ema_momentum_default, name='weights_ema')
            except Exception as e:
              print('Could not apply weights EMA:', e)
      metrics=usersettings.get_metrics(model, loss)

      print('Compiling model...')
      exp_loss_weights=None
      if hasattr(usersettings, 'get_loss_weights'):
        exp_loss_weights=usersettings.get_loss_weights()
      model.compile(optimizer=optimizer,
                    loss=loss,# you can use a different loss on each output by passing a dictionary or a list of losses
                    loss_weights=exp_loss_weights,
                    metrics=metrics) #can specify per output metrics : metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}
      
      if usersettings.enable_mixed_precision:
        tools.gpu.check_mixed_precision_compatibility(model, usersettings)
      
      try:
        init_model_name=os.getcwd()+'/checkpoints/model_init'
        print('******************************************')
        print('Saving the model at init state in ',init_model_name, 'issues at this point should warn you before moving to long training sessions...')
        model.save(init_model_name)
        print('initial model saving done')
        print('******************************************')
        
      except Exception as e:
          raise ValueError('Could not serialize the model, did all model elements defined in the model prior model.compile are serialisable and have their get_config(self) method ? Error message=',e)
  else:#recovering from checkpoint
    print('**** Restoring from checkpoint...')
    import glob
    available_models=sorted(glob.glob(os.path.join(os.getcwd(),'checkpoints/model_epoch_*')), key=os.path.getmtime)
    print('All available model=', available_models)
    loaded_model=available_models[-1]
    print(loaded_model.split('_')[-1])
    initial_epoch=int(loaded_model.split('_')[-1])
    print('Restarting optimization from epoch: '+str(initial_epoch))
    print('loading ',loaded_model)
    model = tf.keras.models.load_model(loaded_model)

  # logs and summaries management:
  #-> classical logging on Tensorboard (scalars, historams, and so on)
  log_dir=os.getcwd()+"/logs/"# + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  #-> specify a summary writer (for more customisation capabilities)
  file_writer = tf.summary.create_file_writer(log_dir)
  file_writer.set_as_default()
  
  # register a specific method to the model to record weights change gradient when manually specifying new weights
  model.track_weights_change = types.MethodType( track_weights_change, model )
  
  # generate the model description
  #-> as an image in the session folder
  model_name_str=usersettings.model_name
  try:
    '''from tensorflow.keras.layers import Layer
    model._layers = [
        layer for layer in model._layers if isinstance(layer, Layer)
    ]'''
    plot_model(model,
              to_file=os.getcwd()+'/'+model_name_str+'.png',
              show_shapes=True)
  except Exception as e:
    print('Could not plot model, error:',e)
  
  #-> as a printed log and write the network summary to file in the session folder
  with open(os.getcwd()+'/'+model_name_str+'.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    print('model.summary', model.summary())
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

  '''try:
    receptive_field_info=tf.contrib.receptive_field.compute_receptive_field_from_graph_def(
                                                          model,
                                                          input_node,
                                                          output_node,
                                                          stop_propagation=None,
                                                          input_resolution=None
                                                      )
  except Exception as e:
    print('Receptive field computation failed, reason=',e)
  '''
  
  if usersettings.debug:
    #TODO to be tested
    print('TODO: check this for tf2 migration...')
    tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    #tf_debug.TensorBoardDebugWrapperSession(tf.Session(), usersettings.debug_server_addresses) #"[[_host]]:[[_port]]")
    '''all_callbacks.append(tf_debug.TensorBoardDebugHook(usersettings.debug_server_addresses,
                                          send_traceback_and_source_code=True,
                                          log_usage=False)
                      )
    '''

  #####################################
  # train the model
  use_multiprocessing=False
  workers=1
  if False:#gpu_workers_nb>1:
    workers*=gpu_workers_nb
    use_multiprocessing=True

  print('Fitting model:')
  print('* use_multiprocessing=',use_multiprocessing)
  if use_multiprocessing:
    print('* multiprocessing workers=',workers)
  history = None

  #manage epoch index in case of fitting interruption/restart
  epoch_start_index=0

  #print XLA mode
  print('*** XLA optimization state : TF_XLA_FLAGS =', os.environ['TF_XLA_FLAGS'])

  #activate tensorflow_model_optimization is required (if tensorflow_model_optimization is available)
  if usersettings.quantization_aware_training:
    try:
      model = tfmot.quantization.keras.quantize_model(model)
      print('Model quantization aware training is being applied, here is quantized model summary')
      print('Have a look at https://www.youtube.com/watch?v=2tnWHCxP8Bk')
      model.summary()
    except Exception as e:
      print('Could not apply model quantization, relying on the original model')
  
  #-> train with (in memory) input data pipelines
  if train_data==val_data:
    raise ValueError('train_data and val_data are the same, please fix this error')
  if usersettings.federated_learning is False or federated_learning_available is False:
    print('Now starting a CENTRALIZED model training session...')
    all_callbacks=define_callbacks(usersettings, model, train_iterations_per_epoch, file_writer, log_dir)
    history = model.fit(
            x=train_data,
            y=None,#train_data,
            batch_size=None,#usersettings.batch_size, #bath size must be specified
            epochs=usersettings.nbEpoch,
            verbose=True,
            callbacks=all_callbacks.values(),
            validation_split=0.0,
            validation_data=val_data,# val_ref),
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=initial_epoch,
            steps_per_epoch=train_iterations_per_epoch,
            validation_steps=val_iterations_per_epoch,
            validation_freq=1,
            max_queue_size=workers*100,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
  else:
    print('Now starting a FEDERATED model training session...')
    # Start Flower client
    federated_learner=federated.FlClient(cid, usersettings, model, train_data, train_iterations_per_epoch, val_data, val_iterations_per_epoch, workers, file_writer, log_dir)
    if 'isServer' not in usersettings.hparams.keys():
      fl.client.start_numpy_client(server_address=usersettings.federated_learning_server_address, client=federated_learner)
      history=federated_learner.history
    else:
      # if function is called for flower simulation, then only the client instance is returned
      return federated_learner
  return history

def run_experiment(usersettings):
  print('Running an experiment....')

  #####################################
  #check GPU requirements vs availability
  if usersettings.debug:
    tf.debugging.set_log_device_placement(True)

  #####################################
  #prepare session
  tf.keras.backend.clear_session() # We need to clear the session to enable JIT in the middle of the program.
  tf.random.set_seed(usersettings.random_seed)

  os.environ['TF_GPU_THREAD_MODE']='gpu_private'
  #(de)activate XLA graph optimization
  tf.config.optimizer.set_jit(usersettings.useXLA)
  if usersettings.useXLA:
    print('Forcing XLA on the CPU side')
    os.environ['TF_XLA_FLAGS']='--tf_xla_cpu_global_jit'
    if usersettings.debug and usersettings.use_profiling:
      os.environ['XLA_FLAGS']='--xla_hlo_profile'

  else:
    os.environ['TF_XLA_FLAGS']=''



  #run a single training experiment
  history=build_run_training_session()

  print('\nhistory dict:', history.history)

  print('Training session end, loss={loss} '.format(loss=history.history['loss'][-1]))
  print('Have a look at the experiments details saved in folder ', os.getcwd())
  final_result=None
  if final_result is None:
    final_result={'loss':history.history['loss'][-1]}
  return final_result, usersettings.model_export_filename


###########################################################
## INFERENCE SECTION : talking to a tensorflow-server
#inspired from https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py

def do_inference(experiment_settings, host, port, model_name, clientIO_InitSpecs, concurrency, num_tests, debug):
  """Tests PredictionService with concurrent requests.
  Args:
    experiment_settings: the experiment settings loaded from function loadExperimentsSettings
    host:tensorfow server address
    port: port address of the PredictionService.
    model_name: the model name ID
    clientIO_InitSpecs: a dictionnary to pass to the ClientIO constructor
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use, infinite prediction loop if <0.
  Raises:
    IOError: An error occurred processing test data set.

  Hint : have a look here to track api use updates : https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py
  """
  stub=model_serving_tools.setup_model_server_connexion(host, port, experiment_settings.grpc_max_message_length)
  #allocate a clientIO instance defined for the experiment
  client_io=experiment_settings.Client_IO(clientIO_InitSpecs, debug)
  notDone=True
  predictionIdx=0
  while notDone:
      # 1. get data and prepare request
      try:
        predictionIdx=predictionIdx+1
        start_time=time.time()
        #get an input data sample to be sent to the model
        sample=client_io.getInputData(predictionIdx)
        request = model_serving_tools.generate_single_request(sample, model_name, debug)
        if debug:
          print('Input data is ready, data=',sample)
          print('Time to prepare collect data request:',round(time.time() - start_time, 2))
      except StopIteration:
        print('End of the process detected, running the ClientIO::finalize method')
        notDone=True
        break

      # 2. send the request
      #asynchronous message request and answer reception, may hide some AbortionError details and only provide CancellationError(code=StatusCode.CANCELLED, details="Cancelled")
      if hasattr(experiment_settings, 'client_async'):
        result_future = stub.Predict.future(request, experiment_settings.serving_client_timeout_int_secs)  # 5 seconds
        result_future.add_done_callback(
            model_serving_tools._create_rpc_callback(client_io, debug))

      else: #synchronous approach
        #synchronous approach... that may provide more details on AbortionError
        if debug:
          start_time=time.time()
        timeout=experiment_settings.serving_client_timeout_int_secs
        if predictionIdx==1:#first request takes longer time (memory allocation, and so on)
          timeout*=5  
        answer=stub.Predict(request, timeout)
        if debug:
          print('Time to send request/decode response:',round(time.time() - start_time, 2))
          start_time=time.time()

        client_io.decodeResponse(answer)
        if debug:
          print('Time to decode response:',round(time.time() - start_time, 2))

      if num_tests>=0:
          if predictionIdx>=num_tests:
              notDone=False
  client_io.finalize()
  return 0

# Run script ##############################################
def run(FLAGS, train_config_script=None, external_hparams={}):
  ''' the main script function that can receive hyperparameters as a dictionnary to run an experiment
  can start training, model serving or model client requesting depending on the FLAGS values:
  -> if FLAGS.start_server is True : starts a server that hosts the target model
  -> if FLAGS.predict is TRUE : starts a client that will send requests to a model threw gRPC
  -> else, start a training session relying on a settings file provided by train_config_script or FLAGS.usersettings
    --> in this mode, function returns a dictionnary of that summarizes the last model states at the end of the training
    --> if calling with non empty train_config_script and with non empty external_hparams,
        then external_hparams will update hyperparameters already specified in the train_config_script script
  '''
  experiments_output=None
  #tf.reset_default_graph()
  usersettings=None#ensure to clear this object prior any new trial
  ''' main function that starts the experiment in the chosen mode '''
  scripts_WD=os.getcwd() #to locate the mysettings*.py file
  print(FLAGS)
  if FLAGS.debug is True:
      print('Running in debug mode. Press Enter to continue...')
  if FLAGS.start_server is True :
      print('### START TENSORFLOW SERVER MODE ###')
      print('WARNING, this function expects some libraries to be installed, mostly dedicated to the training processing.')
      print('-> to run tensorflow model server on minimal install run start_model_serving.py')

      usersettings, sessionFolder = loadExperimentsSettings(os.path.join(FLAGS.model_dir,SETTINGSFILE_COPY_NAME), isServingModel=True)

      #target the served models folder
      model_folder=os.path.join(scripts_WD,FLAGS.model_dir,'exported_models')
      print('Considering served model parent directory:'+model_folder)
      #check if at least one served model exists in the target models directory
      stillWait=True
      while stillWait is True:
        print('Looking for a servable model in '+model_folder)
        try:
          #check served model existance
          if not(os.path.exists(model_folder)):
            raise ValueError('served models directory not found : '+model_folder)
          #look for a model in the directory
          one_model=next(os.walk(model_folder))[1][0]
          one_model_path=os.path.join(model_folder, one_model)
          if not(os.path.exists(one_model_path)):
            raise ValueError('served models directory not found : '+one_model_path)
          print('Found at least one servable model directory '+str(one_model_path))
          stillWait=False
        except Exception as e:
          raise ValueError('Could not find servable model, error='+str(e.message))

      model_serving_tools.get_served_model_info(one_model_path, usersettings.model_name)
      tensorflow_start_cmd=" --port={port} --model_name={model} --model_base_path={model_dir}".format(port=usersettings.tensorflow_server_port,
                                                                                          model=usersettings.model_name,
                                                                                          model_dir=model_folder)
      if len(FLAGS.singularity_tf_server_container_path)>0:
        print('Starting Tensorflow model server from provided singularity container : '+FLAGS.singularity_tf_server_container_path)
        tensorflow_start_cmd='singularity run --nv '+FLAGS.singularity_tf_server_container_path+tensorflow_start_cmd
      else:
        print('Starting Tensorflow model server installed on system')
        tensorflow_start_cmd='tensorflow_model_server '+tensorflow_start_cmd
      print('Starting tensorflow server with command :'+tensorflow_start_cmd)
      os.system(tensorflow_start_cmd)

  elif FLAGS.predict is True or FLAGS.predict_stream !=0:
      print('### PREDICT MODE, interacting with a tensorflow server ###')
      print('If necessary, check the served model behaviors using command line cli : saved_model_cli show --dir path/to/export/model/latest_model/1534610225/ --tag_set serve to get the MODEL_NAME(S)\n to get more details on the target MODEL_NAME, you can then add option --signature_def MODEL_NAME')

      usersettings, sessionFolder = loadExperimentsSettings(os.path.join(FLAGS.model_dir,SETTINGSFILE_COPY_NAME), isServingModel=True)

      #FIXME errors reported on gRPC: https://github.com/grpc/grpc/issues/13752 ... stay tuned, had to install a specific gpio version (pip install grpcio==1.7.3)
      '''server_ready=WaitForServerReady(usersettings.tensorflow_server_address, usersettings.tensorflow_server_port)
      if server_ready is False:
          raise ValueError('Could not reach tensorflow server')
      '''
      print('Prediction mode using model : '+FLAGS.model_dir+' with model '+usersettings.model_name)

      predictions_dir=os.path.join(FLAGS.model_dir,
                              'predictions_'+datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
      os.mkdir(predictions_dir)
      os.chdir(predictions_dir)
      print('Current working directory = '+os.getcwd())
      print('In case of GRPC errors, check codes at https://developers.google.com/maps-booking/reference/grpc-api/status_codes')
      do_inference(
                  experiment_settings=usersettings, host=usersettings.tensorflow_server_address,
                  port=usersettings.tensorflow_server_port,
                  model_name=usersettings.model_name,
                  clientIO_InitSpecs={},
                  concurrency=0,
                  num_tests=FLAGS.predict_stream,
                  debug=FLAGS.debug)

  elif FLAGS.commands is True :
      print('Here are some command examples')
      print('1. train a model (once the mysettings_1D_experiments.py is set):')
      print('-> python experiments_manager.py --usersettings=mysettings_1D_experiments.py')
      print('2. start a tensorflow server on the trained/training model :')
      print('-> python experiments_manager.py --start_server --model_dir=experiments/1Dsignals_clustering/my_test_2018-01-03--14:40:53')
      print('3. interract with the tensorflow server, sending input buffers and receiving answers')
      print('-> python experiments_manager.py --predict --model_dir=experiments/1Dsignals_clustering/my_test_2018-01-03--14\:40\:53/')
      print('4. restart an interrupted training session')
      print('-> python experiments_manager.py --restart_interrupted --model_dir=experiments/1Dsignals_clustering/my_test_2018-01-03--14\:40\:53/')

  else:
      print('### TRAINING MODE, preparing session ###')
      """ This mode first prepares a trainin session, different modes possible:
      -> in classical centralized learning, this script can :
      ------>create and run a new training session in a specific session folder
      ------>restart a training session in an existing specific session folder
      -> in decentralized learning (as for federated learning):
      ------>create the training session folder to be used by the central server
             BUT no training is started, only the session folder location is returned
      """
      """ setting up default values from command line """
      settings_file=FLAGS.usersettings
      """ updating default values if running function from an upper level """

      # manage eventual external custom settings and hyperparameters
      custom_settings=False
      if train_config_script is not None:
        print('--> Training from setup file {file} with the following external hyperparameters {params}'.format(file=train_config_script, params=external_hparams))
        settings_file=train_config_script
        custom_settings=True
      if FLAGS.procID is not None:
        external_hparams['procID']=FLAGS.procID
      if FLAGS.distributed is True:
        external_hparams['distributed']=True
      if len(external_hparams)>0:
        custom_settings=True
        
      if custom_settings:
        print('Some custom settings have been specified : training from setup file {file} with the following external hyperparameters {params}'.format(file=train_config_script, params=external_hparams))
        settings_file=tools.experiments_settings_surgery.insert_additionnal_hparams(settings_file, external_hparams)
        print('-> created a temporary experiments settings file : '+settings_file)



      #loading the experiment setup script
      usersettings, sessionFolder = loadExperimentsSettings(settings_file,
                                                                        restart_from_sessionFolder=FLAGS.model_dir,
                                                                        isServingModel=False)

      #add additionnal hyperparams coming from an optionnal
      if hasattr(usersettings, 'hparams'):
        print('adding hypermarameters declared from the experiment settings script:'+str(usersettings.hparams))
        #update sessionFolder name string
        if not usersettings.recoverFromCheckpoint:
          sessionFolder_splits=sessionFolder.split('_')
          sessionFolder_addon=''
          for key, value in usersettings.hparams.items():
            sessionFolder_addon+='_'+key+str(value)
          #insert sessionname addons in the original one
          sessionFolder=''
          for str_ in  sessionFolder_splits[:-1]:
            sessionFolder+=str_+'_'
          sessionFolder=sessionFolder[:-1]#remove the last '_'
          sessionFolder+=sessionFolder_addon+'_'+sessionFolder_splits[-1]
          usersettings.sessionFolder=sessionFolder
          print('Found hparams: '+str(usersettings.hparams))
      else:
        print('No hparams dictionnary found in the experiment settings file')
      print('Experiment session folder : '+sessionFolder)

      #deduce the experiments settings copy filename that is versionned in the experiment folder
      settings_copy_fullpath=os.path.join(sessionFolder, SETTINGSFILE_COPY_NAME)
      #copy settings and model file to the working folder
      if not usersettings.recoverFromCheckpoint:
        os.makedirs(sessionFolder)
        if hasattr(usersettings, 'model_file'):
          shutil.copyfile(usersettings.model_file, os.path.join(sessionFolder, os.path.basename(usersettings.model_file)))
        settings_src_file=settings_file
        print('Willing to copy {src} to {dst}'.format(src=settings_src_file, dst=settings_copy_fullpath))
        shutil.copyfile(settings_src_file, settings_copy_fullpath)
        #prepare a config file for model serving
        serving_config = configparser.ConfigParser()
        serving_config['SERVER'] = { 'host': usersettings.tensorflow_server_address,
                              'port': usersettings.tensorflow_server_port,
                              'model_name': usersettings.model_name,
                            }
        with open(os.path.join(sessionFolder, 'model_serving_setup.ini'), 'w') as configfile:
          serving_config.write(configfile)
        
      # now ready to start a training session
      # -> for classical/centralised learning, training starts
      # -> if this run function is called from start_feerated_server.py script, then sessionFolder is ready but do not start training at this step
      if 'isFLserver' not in external_hparams.keys():
        # Next keep initial working folder, move to the session folder and run experiment
        initial_wd=os.getcwd()
        os.chdir(sessionFolder)
        res, last_exported_model=run_experiment(usersettings)
        # Finally recover initial working directory
        os.chdir(initial_wd)
        #refactor result in a single updated dictionnary
        experiments_output=res  
        experiments_output.update({'last_exported_model':last_exported_model, 'sessionFolder':sessionFolder})
      else:
        print('SETUP mode, training session folder is prepared and ready to start training.\n No training is started at this step, to be conducted next')
        return sessionFolder
  return experiments_output

if __name__ == "__main__":
  parser=get_commands()
  FLAGS=parser.parse_args()
  run(FLAGS)