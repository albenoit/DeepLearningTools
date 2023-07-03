# ========================================
# FileName: experiment_settings.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of tools to validate the load and check experiments settings file used to train and serve a given model
# for DeepLearningTools.
# =========================================

import os, sys, importlib, datetime
import tensorflow as tf

SETTINGSFILE_COPY_NAME='experiment_settings.py'

def loadModel_def_file(usersettings, absolute_path=False):
  """
  Basic method to load the model targeted by usersettings.model_file.

  :param usersettings: The user (sessionFolder) settings object that contains the model file path.
  :type usersettings: object
  :param absolute_path: Flag indicating whether the model_file path is an absolute path. Defaults to False.
  :type absolute_path: bool, optional
  :return: The loaded Keras model.
  :rtype: keras.Model
  """
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
  """
  Load experiments parameters from the mysettingsxxx.py script.
  Also mask GPUs to only use the ones specified in the settings file.

  :param filename: The settings file. If restarting an interrupted training session, you should target the experiments_settings.py copy available in the experiment folder to restart.
  :type filename: str
  :param call_from_session_folder: Flag indicating whether the function is called from the session folder. Defaults to False.
  :type call_from_session_folder: bool, optional
  :param restart_from_sessionFolder: [OPTIONAL] Set the session folder of a previously interrupted training session to restart.
  :type restart_from_sessionFolder: str, optional
  :param isServingModel: [OPTIONAL] Set True in the case of using model serving (server or client mode) so that some settings are not checked.
  :type isServingModel: bool, optional
  :return: The loaded ExperimentSettings object and the session folder.
  :rtype: Tuple[ExperimentSettings, str]
  """
  
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


class ExperimentSettings(object):
  """
  An experiment settings object with some validators
  """
  def __init__(self, settingsFile, isServingModel, call_from_session_folder=False):
    """
    Loads a python script file that describes an experiment setup and loads its required parameters

    :param settingsFile: The settings filename.
    :type settingsFile: str
    :param isServingModel: Boolean value indicating whether the file is loaded to serve the target model.
    :type isServingModel: bool
    :param call_from_session_folder: Optional parameter to indicate if the method is called from the session folder. Defaults to False.
    :type call_from_session_folder: bool, optional

    :raises ValueError: If the expected parameters are missing or wrong.
    """
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
    self.cid=self.hasOrDefault('cid', '') # enmpty str except in distributed and simulated context, cid indicates the simulated client id and allows subfolders management for logs, checkpoints and expoted_models
    self.hparams_ext=self.hasOrDefault('hparams_ext', {}) #maybe supplementary hyperparameters, not reported in headers
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
    self.federated_learning_server_address=self.hasOrDefault('federated_learning_server_address', "localhost:8080")
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
    self.grpc_max_message_length=self.hasOrDefault('grpc_max_message_length', 536870912, message='specify this parameter if the messages sent to/received from the served model exceed the standard 4Mb size, here using large value (512Mb)')

    #look for an optionnal hyperparameters dictionnary
    self.hparams={} # acollection of external hyperparameters that will decorate session folder name
    if hasattr(self.experiment_settings, 'hparams'):
      self.hparams= getattr(self.experiment_settings,'hparams')
      print('Found hyperparameters:',self.hparams)
      if 'cid' in self.hparams.keys(): #for convenience
         self.cid=self.hparams['cid']
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
    """
    Checks if a parameter exists in the experiment settings object and returns its value if it exists, or a default value if it doesn't.

    :param param: The name of the parameter to check.
    :type param: str
    :param defaultValue: The default value to return if the parameter doesn't exist.
    :type defaultValue: Any
    :param message: Optional message to print when the parameter is not found. Defaults to None.
    :type message: str, optional

    :return: The value of the parameter if it exists, or the default value if it doesn't.
    :rtype: Any
    """
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
    """
    Asserts that a parameter exists in the experiment settings object and its value is positive and above zero.

    :param param: The name of the parameter to check.
    :type param: str
    :param param_description: Description of the parameter for the error message.
    :type param_description: str

    :return: The value of the parameter if it exists and is positive and above zero.

    :raises AssertionError: If the parameter is not found or its value is not positive and above zero.
    """
    message='Specification error on variable {param}: {descr}. It must be set and be above 0'.format(param=param, descr=param_description)
    assert hasattr(self.experiment_settings,param), message
    assert getattr(self.experiment_settings,param)>0, message
    return getattr(self.experiment_settings,param)
  
  def assertPositive_above_equal_zero(self, param, param_description):
    """
    Asserts that a parameter exists in the experiment settings object and its value is positive and above or equal to zero.
    
    :param param: The name of the parameter to check.
    :type param: str
    :param param_description: Description of the parameter for the error message.
    :type param_description: str

    :return: The value of the parameter if it exists and is positive and above or equal to zero.

    :raises AssertionError: If the parameter is not found or its value is not positive and above or equal to zero.
    """
    message='Specification error on variable {param}: {descr}. It must be set and be above 0'.format(param=param, descr=param_description)
    assert hasattr(self.experiment_settings,param), message
    assert getattr(self.experiment_settings,param)>=0, message
    return getattr(self.experiment_settings,param)

  def assertType(self, param, type, param_description):
    """
    Asserts that a parameter exists in the experiment settings object and its value is of the specified type.

    :param param: The name of the parameter to check.
    :type param: str
    :param type: The expected type of the parameter's value.
    :type type: type
    :param param_description: Description of the parameter for the error message.
    :type param_description: str

    :return: The value of the parameter if it exists and is of the specified type.

    :raises AssertionError: If the parameter is not found or its value is not of the specified type.
    """
    message='Specification error on variable {param}: {descr}. It must be of type '.format(param=param, descr=type)
    assert isinstance(getattr(self.experiment_settings,param), type), message
    return getattr(self.experiment_settings,param)

  def has(self, param, error_message):
    """
    Checks if a parameter exists in the experiment settings object.
    
    :param param: The name of the parameter to check.
    :type param: str
    :param error_message: The error message to display if the parameter is missing.
    :type error_message: str

    :return: The value of the parameter if it exists.

    :raises AssertionError: If the parameter is missing in the experiment settings object.
    """
    assert hasattr(self.experiment_settings, param), 'Missing {param} : {msg}'.format(param=param, msg=error_message)
    return getattr(self.experiment_settings,param)

  def summary(self):
    print('******************************************************')
    print('Experiment setup:')
    for key in vars(self).keys():
      print('Setting',key, '=', getattr(self,key))
    print('******************************************************')
