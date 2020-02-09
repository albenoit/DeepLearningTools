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

To adapt to new use case, just duplicate the closest mysettingsxxx file and adjust the configuration.
For any experiment, the availability of all the required fields in the settings file is checked by the experiments_settings_checker.py script.
You can have a look there to ensure you prepared everything right.

As a reminder, here are the functions prototypes:

-define a model to be trained and served in a specific file and follow this prototype:
--report model name in the settings file using variable name model_file or thecify a premade estimator using variable name premade_estimator
--def model( data, #the input data tensor
            hparams,  #external parameters that may be used to setup the model (number of classes and so depending on the task)
            mode), #mode set to switch between train, validate and inference mode
            wrt tf.estimator.tf.estimator.ModeKeys values
          => the model must return a dictionary of output tensors
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

Some examples of such functions are put in the README.md and in the versionned mysettings_xxx.py demos

This demo relies on Tensorflow 1.14 and above and makes use of Estimators
Look at https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/tensorflowcore/trainer/model.py
Look at some general guidelines on Tenforflow here https://github.com/vahidk/EffectiveTensorflow
Look at the related webpages : http://python.usyiyi.cn/documents/effective-tf/index.html
Tensorflow trained graphs can be optimized for inference, some tutorials such as the following may help: https://dato.ml/tensorflow-mobile-graph-optimization/

Glossary : https://developers.google.com/machine-learning/glossary/#custom_estimator
"""

#script imports
from tools.experiment_settings import ExperimentSettings
import tools.experiments_settings_surgery
import os, shutil
import datetime, time
import tensorflow as tf
import numpy as np
import pandas as pd
import importlib
import imp
import copy

try:
    import tensorflow_addons as tfa
except:
    print('WARNING, tensorflow_addons could not be loaded, this may generate errors but should not impact model serving')
#constants
SETTINGSFILE_COPY_NAME='experiment_settings.py'
WEIGHTS_MOVING_AVERAGE_DECAY=0.998

# Set default flags for the output directories
#manage input commands
import argparse
parser = argparse.ArgumentParser(description='demo_semantic_segmentation')
parser.add_argument("-m","--model_dir", default=None,
                    help="Output directory for model and training stats.")
parser.add_argument("-d","--debug", action='store_true',
                    help="set to activate debug mode")
parser.add_argument("-p","--predict", action='store_true',
                    help="Switch to prediction mode")
parser.add_argument("-l","--predict_stream", default=0, type=int,
                    help="set the number of successive predictions, infinite loop if <0")
parser.add_argument("-s","--start_server", action='store_true',
                    help="start the tensorflow server on the machine to run predictions")
parser.add_argument("-psi","--singularity_tf_server_container_path", default='',
                    help="start the tensorflow server on a singularity container to run predictions")
parser.add_argument("-u","--usersettings",
                    help="filename of the settings file that defies an experiment")
parser.add_argument("-r","--restart_interrupted", action='store_true',
                    help="Set to restart an interrupted session, model_dir option should be set")
parser.add_argument("-g","--debug_server_addresses", action='store_true',
                    default="127.0.0.1:2333",
                    help="Set here the IP:port to specify where to reach the tensorflow debugger")
parser.add_argument("-c","--commands", action='store_true',
                    help="show command examples")

FLAGS = parser.parse_args()

class MyCustomModelSaverExporterCallback(tf.keras.callbacks.ModelCheckpoint):
  def __init__(self,
           filepath,
           settings,
           monitor='val_loss',
           verbose=0,
           save_best_only=False,
           save_weights_only=False,
           mode='auto',
           save_freq='epoch',
           **kwargs):
    #init tf.keras.callbacks.ModelCheckpoint parent
    super(MyCustomModelSaverExporterCallback, self).__init__( filepath,
                                                              monitor,
                                                              verbose,
                                                              save_best_only,
                                                              save_weights_only,
                                                              mode,
                                                              save_freq,
                                                              )
    self.settings=settings
    self.settings.model_export_filename=settings.sessionFolder+'/exported_models'
    self.version_count=0

  def on_epoch_end(self, epoch, logs=None):
    #call parent function
    print('Saving checkpoint...')
    super(MyCustomModelSaverExporterCallback, self).on_epoch_end(epoch, logs)

    if self.save_freq == 'epoch':
      try:
        if False:#self.model._in_multi_worker_mode():
          # Exclude training state variables in user-requested checkpoint file.
          with self._training_state.untrack_vars():
            self._export_model(epoch=epoch, logs=logs)
        else:
          self._export_model(epoch=epoch, logs=logs)
      except Exception as e:
        print('Model exporting failed for some reason',e)
    print('Epoch checkpoint save and export processes done.')


  def _export_model(self, epoch, logs=None):
    print('Exporting model...')
    current = logs.get(self.monitor)
    if current==self.best:
      print('EXPORTING A NEW MODEL VERSION FOR SERVING')
      print('Exporting model at epoch {}.'.format(epoch))
      self.version_count+=1
      exported_module=usersettings.get_served_module(self.model, self.settings.model_name)
      if not(hasattr(exported_module, 'served_model')):
        raise ValueError('Experiment settings file MUST have \'served_model\' function with @tf.function decoration.')
      tf.saved_model.save(
        exported_module,
        '{folder}/{version}'.format(folder=self.settings.model_export_filename, version=self.version_count),
        signatures={self.settings.model_name:exported_module.served_model},
        options=None
        )
      print('Model export OK at epoch {}.'.format(epoch))
      older_versions=os.listdir(self.settings.model_export_filename)
      print('Available model versions:',older_versions)
    else:
      print('Model was not exported since no performance increase has been reported')

# Custom generic functions applied when running experiments
def check_GPU_available(usersettings):
  '''check GPU requirements vs availability: if usersettings.used_gpu_IDs is not empty, then check GPU availability accordingly
     Args:
      usersettings, the experiments settings defined by the user
     Raises SystemError if no GPU available
  '''
  gpu_workers_nb=0
  print()
  print('*** GPU devices detection ***')
  # let ensorFlow automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
  tf.config.set_soft_device_placement(True)
  if len(usersettings.used_gpu_IDs)>0:
    device_name = tf.test.gpu_device_name()
    print('Found GPU at: {}'.format(device_name))
    #-> first check availability
    if not tf.test.is_gpu_available():
      #-> first try to wake it up
      os.system("nvidia-modprobe -u -c=0")
      if not tf.test.is_gpu_available():
          raise SystemError('No GPU device found')

    gpus = tf.config.list_physical_devices('GPU')
    print('Found GPU devices:', gpus)
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      visible_devices=[gpus[id] for id in usersettings.used_gpu_IDs]
      print('Setting visible devices:',visible_devices)
      try:
        tf.config.set_visible_devices(visible_devices, 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        #for gpuID in range(len(gpus)):
        #    tf.config.experimental.set_memory_growth(gpus[gpuID], True)
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        gpu_workers_nb=len(logical_gpus)

      except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
  else:
    print('No GPU required for this experiment (usersettings.used_gpu_IDs is empty)')
  return gpu_workers_nb

def loadModel_def_file(sessionFolder):
  ''' basic method to load the model targeted by usersettings.model_file
  Args: sessionFolder, the path to the model file
  Returns: a keras model
  '''
  model_path=os.path.join(sessionFolder,os.path.basename(usersettings.model_file))
  try:
    model_def=imp.load_source('model_def', model_path)#importlib.import_module("".join(model_path.split('.')[:-1]))#
  except Exception as e:
    raise ValueError('loadModel_def_file: Failed to load model file {model} from sessionFolder {sess}, error message={err}'.format(model=usersettings.model_file, sess=sessionFolder, err=e))
  model=model_def.model

  print('loaded model file {file}'.format(file=model_path))
  return model

# Define and run experiment ###############################
def run_experiment(usersettings):
  print('Running an experiment....')

  # import and load tensorflow tools here
  from tensorflow.python import debug as tf_debug
  from tensorflow.keras.models import load_model
  from tensorflow.keras.utils import plot_model
  from tensorboard.plugins.hparams import api as hp

  #####################################
  #check GPU requirements vs availability
  if usersettings.debug:
    tf.debugging.set_log_device_placement(True)
  gpu_workers_nb=check_GPU_available(usersettings)

  #####################################
  #prepare session
  tf.keras.backend.clear_session() # We need to clear the session to enable JIT in the middle of the program.
  tf.random.set_seed(usersettings.random_seed)

  #(de)activate XLA graph optimization
  tf.config.optimizer.set_jit(usersettings.useXLA)
  if usersettings.useXLA:
    print('Forcing XLA on the CPU side')
    os.environ['TF_XLA_FLAGS']='--tf_xla_cpu_global_jit'
    if usersettings.debug and usersettings.use_profiling:
      os.environ['XLA_FLAGS']='--xla_hlo_profile'

  else:
    os.environ['TF_XLA_FLAGS']=''
  #####################################
  # define the input pipepelines (train/val)
  with tf.name_scope('Input_pipeline'):
    train_data =usersettings.get_input_pipeline(raw_data_files_folder=usersettings.raw_data_dir_train,
                                                       isTraining=True)
    val_data = usersettings.get_input_pipeline(raw_data_files_folder=usersettings.raw_data_dir_val,
                                                       isTraining=False)
  try:
    print('Train data=',type(train_data))
    print('Training data samples cardinality=',tf.data.experimental.cardinality(train_data))
    print('Validation data samples cardinality=',tf.data.experimental.cardinality(val_data))
  except Exception as e:
    print('Could not evaluate dataset cardinality', e)
  #####################################
  #create the model from the user defined model file
  # -> (script targeted by usersettings.model_file)
  if usersettings.recoverFromCheckpoint is False:

    print('**** Training from scratch...')
    model_scope=tf.name_scope('model')
    if gpu_workers_nb>1:
      print('Deploying model in a multi/distributed GPU training scheme')
      distribution_strategy=getattr( tf.distribute, usersettings.distribution_strategy)()
      print('-> Chosen distribution strategy :',distribution_strategy)
      model_scope=distribution_strategy.scope()#(model_scope, distribution_strategy.scope())
    usersettings.summary()

    with model_scope:
      #load model
      model=loadModel_def_file(usersettings.sessionFolder)(usersettings)
      #setup training
      learning_rate=usersettings.get_learningRate()
      loss=usersettings.get_total_loss(model)
      optimizer=usersettings.get_optimizer(model, loss, learning_rate)
      if usersettings.weights_moving_averages:
        optimizer=tfa.optimizers.MovingAverage(optimizer, average_decay=0.9999, name='weights_ema')
      metrics=usersettings.get_metrics(model, loss)
      print('Compiling model...')
      model.compile(optimizer=optimizer,
                    loss=loss,# you can use a different loss on each output by passing a dictionary or a list of losses
                    metrics=metrics) #can specify per output metrics : metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}
      try:
          print('Saving the model at init state...')
          model.save(usersettings.sessionFolder+'/checkpoints')
          print('save done')
      except Exception as e:
          raise ValueError('Could not serialize the model, did all model elements defined in the model prior model.compile are serialisable and have their get_config(self) method ? Error message=',e)
  else:#recovering from checkpoint
    print('**** Restoring from checkpoint...')
    model = load_model(usersettings.sessionFolder+'/checkpoints')

  # generate the model description
  #-> as an image in the session folder
  model_name_str=usersettings.model_name
  plot_model(model, to_file=usersettings.sessionFolder+'/'+model_name_str+'.png')

  #-> as a printed log and write the network summary to file in the session folder
  with open(usersettings.sessionFolder+'/'+model_name_str+'.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    print(model.summary())
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
  #####################################
  # prepare all standard callbacks
  all_callbacks=[]

  # -> apply early stopping
  early_stopping_patience=usersettings.early_stopping_patience if hasattr(usersettings, 'early_stopping_patience') else 5
  all_callbacks.append(tf.keras.callbacks.EarlyStopping(
                            monitor='loss',
                            patience=early_stopping_patience
                          )
                        )
  all_callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=(early_stopping_patience*3)//2, min_lr=0.00001, verbose=True)
                        )

  #-> checkpoint each epoch
  all_callbacks.append(MyCustomModelSaverExporterCallback(#tf.keras.callbacks.ModelCheckpoint(
                                            usersettings.sessionFolder+'/checkpoints/',
                                            usersettings,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            save_freq='epoch')
                          )

  if usersettings.debug:
    #TODO to be tested
    print('TODO: check this for tf2 migration...')
    #tf_debug.TensorBoardDebugWrapperSession(tf.Session(), usersettings.debug_server_addresses) #"[[_host]]:[[_port]]")
    '''all_callbacks.append(tf_debug.TensorBoardDebugHook(usersettings.debug_server_addresses,
                                          send_traceback_and_source_code=True,
                                          log_usage=False)
                      )
    '''

  #complete generic callbacks by user defined ones
  all_callbacks+=usersettings.addon_callbacks(model, train_data, val_data)

  #-> classical logging on Tensorboard (scalars, hostorams, and so on)
  log_dir=usersettings.sessionFolder+"/logs/"# + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  #-> activate profiling if required
  profile_batch =0
  if usersettings.use_profiling is True:
    profile_batch = 3
    print('Profiling is applied, for more details and log analysis, check : https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras')
  # -> set embeddings to be logged
  embeddings_layer_names={output.name:output.name+'.tsv' for output in model.outputs}
  print('Model outputs:',embeddings_layer_names)
  #-> set tensorboard logging
  #FIXME: https://github.com/tensorflow/tensorboard/issues/2471
  all_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir,
                                                      histogram_freq=1,
                                                      write_graph=True,
                                                      write_images=False,#True,
                                                      update_freq='epoch',
                                                      profile_batch=profile_batch,
                                                      embeddings_freq=0,
                                                      embeddings_metadata='metadata.tsv'#embeddings_layer_names,
                                                      ))
  #-> add the hyperparameters callback for experimetns comparison
  all_callbacks.append(hp.KerasCallback(log_dir, usersettings.hparams))

  #-> export saved_model (for serving) each epoch

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

  try:
    train_iterations_per_epoch=train_data.n//usersettings.batch_size
    val_iterations_per_epoch=val_data.n//usersettings.batch_size
  except Exception as e:
    print('Could not estimate dataset sizes from input data pipeline, relying on settings nb_train_samples and nb_val_samples.')
    train_iterations_per_epoch=usersettings.nb_train_samples//usersettings.batch_size
    val_iterations_per_epoch=usersettings.nb_val_samples//usersettings.batch_size

  #manage epoch index in case of fitting interruption/restart
  epoch_start_index=0

  #print XLA mode
  print('*** XLA optimization state : TF_XLA_FLAGS =', os.environ['TF_XLA_FLAGS'])

  '''#FIXME: cannot evaluate model.optimizer.iterations to recover epoch index...
  if usersettings.recoverFromCheckpoint:
    epoch_start_index=tf.keras.backend.eval(model.optimizer.iterations)//train_iterations_per_epoch
    print('RESTARTING FITTING FROM CHECKPOINT:')
    print('==> Last training iteration was {lastIt} => setting current epoch to {epoch} '.format(lastIt=model.optimizer.iterations,
                                                                                             epoch=epoch_start_index))
  '''
  print('Now training...')
  #-> train with (in memory) input data pipelines
  history = model.fit(
          x=train_data,
          y=None,#train_data,
          batch_size=None,#usersettings.batch_size, #bath size must be specified
          epochs=usersettings.nbEpoch,
          verbose=True,
          callbacks=all_callbacks,
          validation_split=0.0,
          validation_data=val_data,# val_ref),
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=epoch_start_index,
          steps_per_epoch=train_iterations_per_epoch,
          validation_steps=val_iterations_per_epoch,
          validation_freq=1,
          max_queue_size=workers*100,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
      )

  #else:
  #  raise ValueError('Input data pipeline not supported, should be a tf.data.Dataset or tf.data.generator object')

  # The returned "history" object holds a record
  # of the loss values and metric values during training
  print('\nhistory dict:', history.history)

  print('Training session end, loss={loss} '.format(loss=history.history['loss'][-1]))
  final_result=None
  if final_result is None:
    final_result={'loss':history.history['loss'][-1]}
  return final_result, usersettings.model_export_filename

def train_one_step(model, optimizer, x, y):
  with tf.GradientTape() as tape:
    logits = model(x)
    loss = compute_loss(y, logits)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  compute_accuracy(y, logits)
  return loss

###########################################################
## INFERENCE SECTION : talking to a tensorflow-server
#inspired from https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py

def WaitForServerReady(host, port):
  #inspired from https://github.com/tensorflow/serving/blob/master/tensorflow_serving/model_servers/tensorflow_model_server_test.py
  """Waits for a server on the localhost to become ready.
  returns True if server is ready or False on timeout
  Args:
      host:tensorfow server address
      port: port address of the PredictionService.
  """
  from grpc.beta import implementations
  from grpc.framework.interfaces.face import face
  from tensorflow_serving.apis import predict_pb2
  from tensorflow_serving.apis import prediction_service_pb2
  for _ in range(0, usersettings.wait_for_server_ready_int_secs):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'server_not_real_model_name'

    try:
      # Send empty request to missing model
      print('Trying to reach tensorflow-server {srv} on port {port} for {delay} seconds'.format(srv=host,
                                                             port=port,
                                                             delay=usersettings.wait_for_server_ready_int_secs))
      channel = implementations.insecure_channel(host, int(port))
      stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
      stub.Predict(request, 1)
    except face.AbortionError as error:
      # Missing model error will have details containing 'Servable'
      if 'Servable' in error.details:
        print('Server is ready')
        return True
      else:
        print('Error:'+str(error.details))
    return False
    time.sleep(1)


def _create_rpc_callback(client, debug):
  """Creates RPC callback function.
  Args:
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.
    """
    print('Received response:'+str(result_future))
    exception = result_future.exception()
    if exception:
      #result_counter.inc_error()
      print(exception)
    else:
      try:
          if FLAGS.debug:
              print(result_future.result())
          response=client.decodeResponse(result_future.result())
      except Exception as e:
          raise ValueError('Exception encountered on client callback : '.format(error=e))
  return _callback

def do_inference(experiment_settings, host, port, model_name, clientIO_InitSpecs, concurrency, num_tests):
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
  """
  from tensorflow_serving.apis import predict_pb2 #for single head models
  from tensorflow_serving.apis import inference_pb2 #for multi head models
  from tensorflow_serving.apis import prediction_service_pb2_grpc
  import grpc

  print('Trying to interract with server:{srv} on port {port} for prediction...'.format(srv=host,
                                                         port=port))
  ''' test scripts may help : https://github.com/tensorflow/serving/blob/master/tensorflow_serving/model_servers/tensorflow_model_server_test.py
  '''

  server=host+':'+str(port)
  # specify option to support messages larger than alloed by default
  grpc_options=None
  if usersettings.grpc_max_message_length !=0:
    grpc_options = [('grpc.max_send_message_length', grpc_max_message_size)]
    grpc_options = [('grpc.max_receive_message_length', grpc_max_message_size)]
  channel = grpc.insecure_channel(server, options=grpc_options)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  #allocate a clientIO instance defined for the experiment
  client_io=experiment_settings.Client_IO(clientIO_InitSpecs, FLAGS.debug)
  notDone=True
  predictionIdx=0
  while notDone:
      try:
        predictionIdx=predictionIdx+1
        start_time=time.time()
        sample=client_io.getInputData(predictionIdx)
        if not(isinstance(sample, dict)):
          raise ValueError('Expecting a dictionnary of values that will further be converted to proto buffers. Dictionnary keys must correspond to the usersettings.served_input_names strings list')
        if FLAGS.debug:
            print('Input data is ready, data=',sample)
            print('Time to prepare collect data request:',round(time.time() - start_time, 2))
            start_time=time.time()
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.signature_name = model_name#experiment_settings.served_head_names[0]
        for inputname in usersettings.served_input_names:
          #print('input name:',inputname)
          feature=sample[inputname]
          feature_proto=tf.make_tensor_proto(feature, shape=feature.shape)
          #print('->feature:',sample[inputname])
          #print('->proto:',feature_proto)
          request.inputs[inputname].CopyFrom(feature_proto)
        if FLAGS.debug:
          print('Time to prepare request:',round(time.time() - start_time, 2))
      except StopIteration:
        print('End of the process detection, running the ClientIO::finalize method')
        notDone=True
        break

      #asynchronous message reception, may hide some AbortionError details and only provide CancellationError(code=StatusCode.CANCELLED, details="Cancelled")
      if hasattr(experiment_settings, 'client_async'):
        result_future = stub.Predict.future(request, experiment_settings.serving_client_timeout_int_secs)  # 5 seconds
        result_future.add_done_callback(
            _create_rpc_callback(client_io, FLAGS.debug))

      else:
        #synchronous approach... that may provide more details on AbortionError
        if FLAGS.debug:
          start_time=time.time()
        answer=stub.Predict(request, experiment_settings.serving_client_timeout_int_secs)
        if FLAGS.debug:
          print('Time to send request/decode response:',round(time.time() - start_time, 2))
          start_time=time.time()

        client_io.decodeResponse(answer)
        if FLAGS.debug:
          print('Time to decode response:',round(time.time() - start_time, 2))

      if num_tests>=0:
          if predictionIdx>=num_tests:
              notDone=False
  client_io.finalize()
  return 0

def loadExperimentsSettings(filename, restart_from_sessionFolder=None, isServingModel=False):
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

    usersettings=ExperimentSettings(filename, isServingModel)

    if isServingModel:
      sessionFolder=os.path.dirname(filename)

    #manage the working folder in the case of a new experiment
    workingFolder=usersettings.workingFolder
    if restart_from_sessionFolder is None:
      sessionFolder=os.path.join(workingFolder, usersettings.session_name+'_'+datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
    print('Considered usersettings.hparams=',str(usersettings.hparams))
    return usersettings, sessionFolder

def get_served_model_info(one_model_path, expected_model_name):
  ''' basic function that checks served model behaviors
  Args:
  one_model_path: the path to a servable model directory
  expected_model_name: the model name that is expected to be found on the server
  Returns:
    Nothing for now
  '''
  import subprocess
  #get the first subfolder of the served models directory
  served_model_info_cmd='saved_model_cli show --dir {target_model} --tag_set serve --signature_def {model_name}'.format(target_model=one_model_path,
                                                                                      model_name=expected_model_name)
  print('Checking served model available signatures using command '+served_model_info_cmd)
  cmd_result=subprocess.check_output(served_model_info_cmd.split())
  print('Answer:')
  print(cmd_result.decode())
  if expected_model_name in cmd_result.decode():
    print('Target model {target} name found in the command answer'.format(target=expected_model_name))
  else:
    raise ValueError('Target model {target} name NOT found in the command answer'.format(target=expected_model_name))

# Run script ##############################################
def run(train_config_script=None, external_hparams=None):
  ''' the main script function that can receive hyperparameters as a dictionnary to run an experiment
  can start training, model serving or model client requesting depending on the FLAGS values:
  -> if FLAGS.start_server is True : starts a server that hosts the target model
  -> if FLAGS.predict is TRUE : starts a client that will send requests to a model threw gRPC
  -> else, strat a training session relying on a sesstings file provided by train_config_script or FLAGS.usersettings
    --> in this mode, function returns a dictionnary of that summarizes the last model states at the end of the training
    --> if calling with non empty train_config_script and with non empty external_hparams,
        then external_hparams will update hyperparameters already specified in the train_config_script script
  '''
  global usersettings
  experiments_output=None
  #tf.reset_default_graph()
  usersettings=None#ensure to clear this object prior any new trial
  ''' main function that starts the experiment in the chosen mode '''
  scripts_WD=os.getcwd() #to locate the mysettings*.py file

  if FLAGS.debug is True:
      print('Running in debug mode. Press Enter to continue...')
  if FLAGS.start_server is True :
      print('### START TENSORFLOW SERVER MODE ###')

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

      get_served_model_info(one_model_path, usersettings.model_name)
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
      do_inference(experiment_settings=usersettings, host=usersettings.tensorflow_server_address,
                  port=usersettings.tensorflow_server_port,
                  model_name=usersettings.model_name,
                  clientIO_InitSpecs={},
                  concurrency=0,
                  num_tests=FLAGS.predict_stream)

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
      print('### TRAINING MODE ###')
      """ setting up default values from command line """
      settings_file=FLAGS.usersettings
      """ updating default values if running function from an upper level """

      if train_config_script!=None:
        print('Non command line mode : training from setup file {file} with the following external hyperparameters {params}'.format(file=train_config_script, params=external_hparams))
        settings_file=tools.experiments_settings_surgery.insert_additionnal_hparams(train_config_script, external_hparams)
        print('-> created a temporary experiments settings file : '+settings_file)
      #loading the experiment setup script
      usersettings, sessionFolder = loadExperimentsSettings(settings_file,
                                                                        restart_from_sessionFolder=FLAGS.model_dir,
                                                                        isServingModel=False)

      #update hparams structure with external parameters
      usersettings.debug=FLAGS.debug
      usersettings.debug_server_addresses=FLAGS.debug_server_addresses
      #add additionnal hyperparams coming from an optionnal
      if hasattr(usersettings, 'hparams'):
        print('adding hypermarameters declared from the experiment settings script:'+str(usersettings.hparams))
        #update sessionFolder name string
        if not FLAGS.restart_interrupted:
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
      usersettings.sessionFolder=sessionFolder
      if not FLAGS.restart_interrupted:
        os.makedirs(sessionFolder)
        if hasattr(usersettings, 'model_file'):
          shutil.copyfile(usersettings.model_file, os.path.join(sessionFolder, os.path.basename(usersettings.model_file)))
        settings_src_file=settings_file
        print('Willing to copy {src} to {dst}'.format(src=settings_src_file, dst=settings_copy_fullpath))
        shutil.copyfile(settings_src_file, settings_copy_fullpath)
      else:
        usersettings.recoverFromCheckpoint=True
      res, last_exported_model=run_experiment(usersettings)
      #refactor result in a single updated dictionnary
      experiments_output=res
      experiments_output.update({'last_exported_model':last_exported_model, 'sessionFolder':sessionFolder})
  return experiments_output

if __name__ == "__main__":
    run()
