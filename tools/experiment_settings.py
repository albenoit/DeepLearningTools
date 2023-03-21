'''
@author : Alexandre Benoit, LISTIC lab, FRANCE
@brief  : a set of tools to validate the experiments settings file used to train and serve a given model
'''

import os, sys, importlib
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


class MyCustomModelSaverExporterCallback(tf.keras.callbacks.ModelCheckpoint):
  def __init__(self,
           filepath,
           settings,
           monitor='val_loss',
           verbose=1,
           save_best_only=False,
           save_weights_only=False,
           mode='auto',
           save_freq='epoch',
           previous_model_params=None,
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
    self.settings.model_export_filename=os.path.join(os.getcwd(),'exported_models')
    self.previous_model_params=previous_model_params

  def on_epoch_end(self, epoch, logs=None):
    #call parent function
    print('on_epoch_end, saving checkpoint...')
    super(MyCustomModelSaverExporterCallback, self).on_epoch_end(epoch, logs)

    #log model change wrt previous epoch
    if self.previous_model_params !=None:
      self.model.track_weights_change(self.previous_model_params, epoch, prefix='on_epoch')
    self.previous_model_params=self.model.get_weights()
    if logs is None:
      print('WARNING, no logs dict is provided to ModelCheckpoint.on_epoch_end, checkpointing on best epoch will not work')
    
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
      print('Saving complete keras model thus enabling fitting restart as well as model load and predict')
      checkpoint_folder=os.path.join(os.getcwd(),'checkpoints/model_epoch_{version}'.format(version=epoch))
      self.model.save(checkpoint_folder)
      print('EXPORTING A NEW MODEL VERSION FOR SERVING')
      print('Exporting model at epoch {}.'.format(epoch))
      exported_module=self.settings.get_served_module(self.model, self.settings.model_name)
      if not(hasattr(exported_module, 'served_model')):
        raise ValueError('Experiment settings file MUST have \'served_model\' function with @tf.function decoration.')
      if self.settings.save_only_last_best_model:
        output_path='{folder}/1'.format(folder=self.settings.model_export_filename)
      else:
        output_path='{folder}/{version}'.format(folder=self.settings.model_export_filename, version=epoch)
      
      model_concrete_function=exported_module.served_model.get_concrete_function()
      signatures={'serving_default':model_concrete_function, self.settings.model_name:model_concrete_function}
      try:
        module_graph= model_concrete_function
        print(module_graph.pretty_printed_signature(verbose=True))
          
        #module_graph[self.settings.model_name]=exported_module.served_model
        print('Exporting RAW serving model relying on tf.saved_model.save')
        tf.saved_model.save(
          exported_module,
          output_path,
          signatures=signatures,
          options=None
          )
        with open(self.settings.model_export_filename+'/modelserving.signatures', 'w') as f:
          f.write(module_graph.pretty_printed_signature())  
    
        if len(self.settings.used_gpu_IDs)==0:
          print('No GPU available to enable model export with TF-TensorRT')
        else:
          print('Now exporting model for inference with TensorRT...')
          try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            print(os.listdir(output_path))
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
            if self.settings.enable_mixed_precision:
              #TODO, refine precision conversion, see : https://colab.research.google.com/github/vinhngx/tensorrt/blob/vinhn-tf20-notebook/tftrt/examples/image-classification/TFv2-TF-TRT-inference-from-Keras-saved-model.ipynb?hl=en#scrollTo=qKSJ-oizkVQY
              # and official doc https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#tf-trt-api-20
              conversion_params = conversion_params._replace(precision_mode="FP16")
            print('Export for inference conversion_params:', conversion_params)
            converter = trt.TrtGraphConverterV2(input_saved_model_dir=output_path)
            converter.convert()
            converter.summary()
            converter.save(output_path+'trt')
            print('Export for inference OK')
          except Exception as e:
            print('Failed to export sstandard serving model with model.save approach. Reported error message:', e)
            
        print('Exporting to TFLite...')
        try:
          converter = tf.lite.TFLiteConverter.from_saved_model(output_path)
          #converter = tf.lite.TFLiteConverter.from_keras_model(exported_module)
          #converter.optimizations = [tf.lite.Optimize.DEFAULT]
          quantized_tflite_model = converter.convert()
          with open(output_path+'saved_model.tflite', 'wb') as f:
            f.write(quantized_tflite_model)

          # Save the optimized graph'test.pb'
          tf.io.write_graph(graph_or_graph_def=quantized_tflite_model,
                            logdir= output_path,
                            name= 'saved_model.tflite',
                            as_text=False) 
          
          print('Export to TFLite OK')
        except Exception as e:
          print('Could not export model to tflite, reported error:', e)
      except Exception as e:
        print('Failed to export standard serving model with model.save approach. Reported error message:', e)
        
      
      print('Model export OK at epoch {}.'.format(epoch))
      older_versions=os.listdir(self.settings.model_export_filename)
      print('Available model versions:',older_versions)
    else:
      print('Model was not exported since no performance increase has been reported')

class CustomHistory(tf.keras.callbacks.History):
  """Callback that records events into a `History` object.
  This callback is automatically applied to
  every Keras model. The `History` object
  gets returned by the `fit` method of models.
  """

  def __init__(self):
    super(CustomHistory, self).__init__()
    self.history = {}
    self.epoch = [] #moved from on_train_begin, helps keeping the same  log for multiple training sessions

  def on_train_begin(self, logs=None):
    print('******** HISTORY starting a training session...')
  
  def on_epoch_end(self, epoch, logs=None):
    print('******** HISTORY on_epoch_end...')

    super(CustomHistory, self).on_epoch_end(epoch, logs)

#####################################
# prepare all standard callbacks as a dictionary
def define_callbacks(usersettings, model, train_iterations_per_epoch, file_writer, log_dir, previous_model_params=None):
  all_callbacks={}

  #add the history callback
  all_callbacks['history_callback']=CustomHistory()#tf.keras.callbacks.History()
  # -> terminate on NaN loss values
  all_callbacks['TerminateOnNaN_callback']=tf.keras.callbacks.TerminateOnNaN()
  # -> apply early stopping
  early_stopping_patience=usersettings.early_stopping_patience if hasattr(usersettings, 'early_stopping_patience') else 5
  all_callbacks['earlystopping_callback']=tf.keras.callbacks.EarlyStopping(
                            monitor=usersettings.monitored_loss_name,
                            patience=early_stopping_patience,
                            restore_best_weights=True
                          )
  all_callbacks['reduceLROnPlateau_callback']=tf.keras.callbacks.ReduceLROnPlateau(monitor=usersettings.monitored_loss_name, factor=0.1,
                              patience=(early_stopping_patience*2)//3, min_lr=0.000001, verbose=True)

  #-> checkpoint each epoch
  all_callbacks['checkpoint_callback']=MyCustomModelSaverExporterCallback(#tf.keras.callbacks.ModelCheckpoint(
                                            os.path.join(os.getcwd(),'checkpoints/'),
                                            usersettings,
                                            monitor=usersettings.monitored_loss_name,
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            save_freq='epoch',
                                            previous_model_params=previous_model_params)


  if callable(usersettings.custom_tensorboard_logs):
    print('Adding custom Tensorboard logger ON EPOCH END. keep some time, have a look at existing tools in tools/custom_display_tensorboard.py')
    with file_writer.as_default():
      def custom_logger_fn(epoch, logs):
        usersettings.custom_tensorboard_logs(model, epoch, logs)
      all_callbacks['on_epoch_end_custom_logger_callback']=tf.keras.callbacks.LambdaCallback(on_epoch_end=custom_logger_fn)
  #complete generic callbacks by user defined ones
  #all_callbacks+=usersettings.addon_callbacks(model, train_data, val_data)

  #-> activate profiling if required
  profile_batch =0
  if usersettings.use_profiling is True:
    print('train_iterations_per_epoch', train_iterations_per_epoch)
    t_start=int(max(1,train_iterations_per_epoch//2-30))
    t_stop=int(t_start+min(60, train_iterations_per_epoch//4))
    if t_stop>train_iterations_per_epoch:
      t_stop=train_iterations_per_epoch-1
      print('too few iterations to perform a reliable profiling')
    profile_batch = '{t1},{t2}'.format(t1=t_start, t2=t_stop)
    print('Profiling is applied, for more details and log analysis, check : https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras')
  # -> set embeddings to be logged
  embeddings_layer_names={output.name:output.name+'.tsv' for output in model.outputs}
  print('Model outputs:',embeddings_layer_names)
  #-> set tensorboard logging
  #FIXME: https://github.com/tensorflow/tensorboard/issues/2471
  all_callbacks['tensorboard_callback']=tf.keras.callbacks.TensorBoard(log_dir,
                                                      histogram_freq=1,
                                                      write_graph=True,
                                                      update_freq='epoch',
                                                      profile_batch=profile_batch,
                                                      embeddings_freq=10,
                                                      )
                                                      #embeddings_metadata='metadata.tsv',
                                                      #embeddings_layer_names=list(embeddings_layer_names.keys()),#'embedding',
                                                      #embeddings_data='stuff'
                                                      #)
  #-> add the hyperparameters callback for experiments comparison
  all_callbacks['hyperparam_callback']=hp.KerasCallback(log_dir, usersettings.hparams)
  return all_callbacks

class ExperimentSettings(object):
  '''an experiment settings object with some validators
  '''
  def __init__(self, settingsFile, isServingModel, call_from_session_folder=False):
    ''' loads a python script file that describes an experiment setup
    and loads its required parameters
      Args:
        settingsFile, the settings filename
        isServingModel, boolean, True if loading file to serve the target model
      Raises:
        ValueError if the expected parameters are missing or wrong
    '''
    # load the settings file
    print('Trying to load experiments settings file : '+str(settingsFile))
    try:
        settingsFile=os.path.normpath(settingsFile) #convert to standard path with single '/' separators

        module_spec = importlib.util.spec_from_file_location('settings', settingsFile)
        self.experiment_settings = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self.experiment_settings)

        '''settings_module=settingsFile[:-3].split('/')
        if len(settings_module)>1:
          settingsFolder=os.path.join(*settings_module[:-1]).replace('/','.')
          settings_module=settings_module[-1]
          settings_module=settingsFolder+'.'+settings_module
        else:# as for serve and predict mode, working directory is expected to be the experiment folder
          settings_module=settingsFile[:-3]
        print('-> loading as module : '+settings_module)
        print(os.getcwd())
        print(os.listdir())
        print('settings file exists:',os.path.exists(settings_module+'.py'))
        self.experiment_settings = importlib.import_module(settings_module)
        '''
    except Exception as e:
        raise ValueError('Failed to load {settings} file. Error message is {error}. This generally comes from a) file does not exist, b)basic python syntax errors'.format(settings=settingsFile, error=e))

    #experiment session parameters
    print('******************************************************')
    print('* Checking the experiments settings file...')
    print('* look at the README.md file to read a working example')
    print('* look at the experiment_settings_checker script to see all the required fields')

    #experiment session parameters
    self.random_seed=self.assertType('random_seed', int, 'an integer to specify the random seed to apply')
    self.recoverFromCheckpoint=self.hasOrDefault('recoverFromCheckpoint', False)
    self.sessionFolder=''
    self.session_name=self.assertType('session_name', str, 'a string that specifies the name of the experiment')
    self.workingFolder=self.assertType('workingFolder', str, 'a string that specifies the parent pathname where the training procedure data is being stored')
    self.debug=self.hasOrDefault('debug', False)
    self.debug_server_addresses=self.hasOrDefault("debug_server_addresses", "127.0.0.1:2333")
    self.model_name=''
    self.useXLA=self.assertType('useXLA', bool, 'activate or not XLA graph optimization, must be boolean')
    self.model_export_filename=''
    self.weights_moving_averages=self.assertType('weights_moving_averages', bool, 'weights_moving_averages must be boolean')
    self.enable_mixed_precision=self.hasOrDefault('enable_mixed_precision', False)
    #model fitting parameters
    self.batch_size=self.assertPositive_above_zero('batch_size', 'the number of samples processed for each batch')
    self.nbEpoch=self.assertPositive_above_zero('nbEpoch', 'the number of times the training set is processed for training')
    self.verbose=1
    self.class_weight=[]
    self.sample_weight=[]
    self.initial_epoch=0
    self.validation_freq=1
    self.max_queue_size=10
    self.workers=1
    self.use_multiprocessing=False
    self.quantization_aware_training=self.hasOrDefault('quantization_aware_training', False)
    self.federated_learning=self.hasOrDefault('enable_federated_learning', False)
    self.federated_learning_server_address=self.hasOrDefault('federated_learning_server_address', "[::]:8080")
    # learning rate management
    self.early_stopping_patience=self.assertPositive_above_zero('early_stopping_patience', 'the number of epoch without val_loss decrease to wait for before stopping training')
    self.get_learningRate=self.has('get_learningRate', 'the training speed factor, returns a float or a tf.keras.optimizers.schedules object')
    self.monitored_loss_name=self.hasOrDefault('monitored_loss_name', 'val_loss')
    #input data parameters
    self.raw_data_dir_val=self.has('raw_data_dir_val', 'path to the validation dataset')
    self.raw_data_dir_train=self.has('raw_data_dir_train', 'path to the training dataset')
    self.raw_data_filename_extension=self.assertType('raw_data_filename_extension', str, 'extension of the dataset files')
    self.consume_data_from_kafka=self.hasOrDefault('enable_federated_learning', False)
    #GPU IDs allocation
    self.used_gpu_IDs=self.assertType('used_gpu_IDs', list, 'a list of GPU ids to use for the experiment')
    self.distribution_strategy=self.hasOrDefault('distribution_strategy', 'MirroredStrategy')
    #profiling
    self.use_profiling=self.assertType('use_profiling', bool, 'Boolean to ativate processing workflow profiling')
    #check train and validation dataset parameters
    self.nb_train_samples=self.assertPositive_above_equal_zero('nb_train_samples', 'the number of samples used for training')
    self.nb_val_samples=self.assertPositive_above_equal_zero('nb_val_samples', 'the number of samples used for validation')
    self.kafka_bootstrap_servers=self.hasOrDefault('kafka_bootstrap_servers', ['localhost:9092'])
    self.consume_data_from_kafka=self.hasOrDefault('consume_data_from_kafka', False)
    #input pipelines
    self.get_input_pipeline=self.has('get_input_pipeline', 'the train and validation input data pipelines function params=[batch_size, raw_data_files_folder, shuffle_batches], must return an input function as described here : https://www.tensorflow.org/programmers_guide/datasets')
    self.custom_tensorboard_logs=self.hasOrDefault('custom_tensorboard_logs', False, message='specify a function able to add some tf.summary.x (x could be image, etc.')
    #tensorflow serving and client dialog
    self.save_only_last_best_model=self.hasOrDefault('save_only_last_best_model', True, message='set True in order to only save the last best model in the exported model folder (keep disk space)')
    self.wait_for_server_ready_int_secs=self.assertPositive_above_zero('wait_for_server_ready_int_secs', 'the number of seconds to wait for a tensorflow service before timeout on first contact')
    self.serving_client_timeout_int_secs=self.assertPositive_above_zero('serving_client_timeout_int_secs', 'the number of seconds to wait for a tensorflow service before timeout for each request')
    self.served_head_names=self.assertType('served_head_names', list, 'a list string(s) providing the name(s) of the target output, relates to the get_served_module function behaviors')
    self.served_input_names=self.assertType('served_input_names', list, 'a list of string(s) providing the name(s) of the input(s) of the served model, relates to the get_served_module function behaviors')
    self.Client_IO=self.has('Client_IO', 'a Client_IO class that defines how a client talks to a tensorflow server')
    self.tensorflow_server_address=self.has('tensorflow_server_address', 'a string specifying the IP adress of the tensorflow server to be contacted by a client')
    self.tensorflow_server_port=self.has('tensorflow_server_port', 'an integer that specifies the port use to communicate whith the tensorflow server')
    self.get_served_module=self.has('get_served_module', 'a function that returns a @tf.function that specifies how to use the model in production/serving')
    self.grpc_max_message_length=self.hasOrDefault('grpc_max_message_length', 0, message='specify this parameter if the messages sent to/received from the served model exceed the standard 4Mb size')

    #look for an optionnal hyperparameters dictionnary
    self.hparams={} # acollection of external hyperparameters that will decorate session folder name
    if hasattr(self.experiment_settings, 'hparams'):
      self.hparams= getattr(self.experiment_settings,'hparams')
      print('Found hyperparameters:',self.hparams)

    #check model
    self.model_file=os.path.normpath(self.assertType('model_file', str, 'model_file must be set as a filename targetting the model description to optimise'))
    #since model file is expected to be in the current directory, get the basename
    if (call_from_session_folder):
       self.model_file=os.path.basename(self.model_file)
    # #-> set model_name accordingly
    self.model_name=self.__get_model_name()

    if isServingModel is False:
      assert os.path.exists(self.model_file), '{model} targetted by model_file filename does not exist in the current working directory: {cwd}'.format(model=self.model_file, cwd=os.getcwd())

    #train and validation flags and functions
    self.get_total_loss=self.has('get_total_loss', 'function that receives parameters [inputs, model_outputs_dict, labels, weights_loss] that must return a graph node that represents the optimisation loss. This node will be drawn on the tensorboard as the \'loss\' variable.')
    self.get_metrics=self.has('get_metrics', 'function that receives parameters [inputs, model_outputs_dict, labels] and that returns a dictionnary of tf.metrics')
    self.get_optimizer=self.has('get_optimizer', 'function that receives as parameters [loss, learning_rate, global_step] and that outputs an optimisation node (generally a loss minimization op)')
    self.reference_labels=self.assertType('reference_labels', list, 'a list of strings THAT MUST BE of the same length as the number of model output tensors')
    self.addon_callbacks=self.has('addon_callbacks', 'a function that receives tf.keras.model and train and val input pipelines as parameters and retruns a list of tf.keras.callbacks ')


    for key in vars(self).keys():
      assert getattr(self,key)!=None, 'Parameter '+key+' not initialized'

    print('INFO: All required parameters are set')

    print('*** Parameters check ended successfuly')
    print('******************************************************')

  def hasOrDefault(self, param, defaultValue, message=None):
      if hasattr(self.experiment_settings,param):
        return getattr(self.experiment_settings,param)
      else:
        print('Did not find parameter', param, 'using as default', str(defaultValue))
        if message is not None:
          print('->',message)
        return defaultValue

  def __get_model_name(self):
          return self.model_file.split('.')[0].split('/')[-1]

  def assertPositive_above_zero(self, param, param_description):
          message='Specification error on variable {param}: {descr}. It must be set and be above 0'.format(param=param, descr=param_description)
          assert hasattr(self.experiment_settings,param), message
          assert getattr(self.experiment_settings,param)>0, message
          return getattr(self.experiment_settings,param)
  def assertPositive_above_equal_zero(self, param, param_description):
          message='Specification error on variable {param}: {descr}. It must be set and be above 0'.format(param=param, descr=param_description)
          assert hasattr(self.experiment_settings,param), message
          assert getattr(self.experiment_settings,param)>=0, message
          return getattr(self.experiment_settings,param)

  def assertType(self, param, type, param_description):
          message='Specification error on variable {param}: {descr}. It must be of type '.format(param=param, descr=type)
          assert isinstance(getattr(self.experiment_settings,param), type), message
          return getattr(self.experiment_settings,param)

  def has(self, param, error_message):
      assert hasattr(self.experiment_settings, param), 'Missing {param} : {msg}'.format(param=param, msg=error_message)
      return getattr(self.experiment_settings,param)

  def summary(self):
    print('******************************************************')
    print('Experiment setup:')
    for key in vars(self).keys():
      print('Setting',key, '=', getattr(self,key))
    print('******************************************************')
