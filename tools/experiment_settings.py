'''
@author : Alexandre Benoit, LISTIC lab, FRANCE
@brief  : a set of tools to validate the experiments settings file used to train and serve a given model
'''

import os, sys, importlib

class ExperimentSettings(object):
  '''an experiment settings object with some validators
  '''
  def __init__(self, settingsFile, isServingModel):
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
    self.recoverFromCheckpoint=False
    self.sessionFolder=''
    self.session_name=self.assertType('session_name', str, 'a string that specifies the name of the experiment')
    self.workingFolder=self.assertType('workingFolder', str, 'a string that specifies the parent pathname where the training procedure data is being stored')
    self.debug=False
    self.debug_server_addresses="127.0.0.1:2333"
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
    #-> set model_name accordingly
    self.model_name=self.__get_model_name()

    if isServingModel is False:
      assert os.path.exists(self.model_file), '{model} targetted by model_file filename does not exist'.format(model=getattr(self.experiment_settings, 'model_file'))

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
