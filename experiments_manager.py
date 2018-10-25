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


#Machine Setup (tested with tensorflow from 1.4.1 to 1.8)
1. install python 2.7 and python pip
2. install tensorflow and tensorflow serving using pip : pip install tensorflow-gpu tensorflow-serving-api
Note that the first versions of the dependency lib grpcio may bring some troubles when starting the tensorflow server.
grpcio python library version 1.7.3 and latest version above 1.8.4 should work.
==> Additionnal recommendations:
Get much better performances with optimized tensorflow packages coming from here:
https://github.com/mind/wheels/releases/
Install like this adajust the last link to your target version:
pip install --ignore-installed --upgrade \ https://github.com/mind/wheels/releases/download/tf1.4.1-gpu-cuda9/tensorflow-1.4.1-cp27-cp27mu-linux_x86_64.whl
Get the Intel MKL library installed :
https://github.com/mind/wheels#mkl


#How tu use it ?

The main script is experiments_manager.py can be used in 3 modes, here are some command examples:
1. train a model in a context specified in a parameters script such as mysettings_1D_experiments.py:
-> python experiments_manager.py --usersettings=mysettings_1D_experiments.py
2. start a tensorflow server on the trained/training model :
-> python experiments_manager.py --start_server --model_dir=experiments/1Dsignals_clustering/my_test_2018-01-03--14:40:53
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
-def getOptimizer(loss, model_outputs_dict, learning_rate, global_step)
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

This demo relies on Tensorflow 1.7 and above and makes use of Estimators
Look at https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/tensorflowcore/trainer/model.py
Look at some general guidelines on Tenforflow here https://github.com/vahidk/EffectiveTensorflow
Look at the related webpages : http://python.usyiyi.cn/documents/effective-tf/index.html
Tensorflow trained graphs can be optimized for inference, some tutorials such as the following may help: https://dato.ml/tensorflow-mobile-graph-optimization/

Glossary : https://developers.google.com/machine-learning/glossary/#custom_estimator
"""

from experiments_settings_checker import ExperimentsSettingsChecker
import os, shutil
import datetime, time
import tensorflow as tf
import numpy as np
import pandas as pd
import imp
import copy
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.contrib import slim
from tensorflow.python import debug as tf_debug

global usersettings
embeddingsFolder='embeddings'
settingsFile_saveName='experiment_settings.py'

MOVING_AVERAGE_DECAY=0.998
# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG)

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_string("settings_file",FLAGS.usersettings,"settings file to load")
tf.app.flags.DEFINE_string ('model_dir', None,'Output directory for model and training stats.')
tf.app.flags.DEFINE_boolean("debug",False,"activate debug information display (ops device placement, some buffer sizes, etc.)")
tf.app.flags.DEFINE_boolean("predict", False, "Switch to prediction mode")
tf.app.flags.DEFINE_boolean("start_server",False,"start the tensorflow server on the machine to run predictions")
tf.app.flags.DEFINE_boolean("commands",False, "Display some command examples")
tf.app.flags.DEFINE_string ("usersettings",'mysettings_1D_experiments.py', "filename of the settings file dedicated to some experiment(s)")
tf.app.flags.DEFINE_integer("predict_stream",0,"this value number of predictions, infinite loop if <0")
tf.app.flags.DEFINE_boolean("restart_interrupted", False, "Set True to restart an interrupted session, model_dir option should be set")
tf.app.flags.DEFINE_string ("debug_server_addresses", "127.0.0.1:2333", "Set here the IP:port to specify where to reach the tensorflow debugger")

def loadModel(sessionFolder):
  ''' basic method to load the model targeted by usersettings.model_file
  '''
  model_path=os.path.join(sessionFolder,usersettings.model_file)
  try:
    model_def=imp.load_source('model_def', model_path)
  except Exception,e:
    raise ValueError('loadModel: Failed to load model file {model} from sessionFolder {sess}, error message={err}'.format(model=usersettings.model_file, sess=sessionFolder, err=e))
  model=model_def.model

  print('loaded model file {file}'.format(file=model_path))
  return model

def getIterationsPerEpoch(mode):
  ''' given a mode (train or test), compute the number of iterations required to parse the related dataset (one epoch)
  Args:
     mode: a string, 'train' or 'val' to set which dataset to consider
  Returns:
     an integer : the number of iterations required to do an epoch
  '''
  nbSamples=0
  #get the number of samples wrt target dataset
  if mode == 'train':
    nbSamples=usersettings.nb_train_samples
  elif mode=='val':
    nbSamples=usersettings.nb_test_samples
  else:
    raise ValueError('Expected parameter string \'train\' or \'val\' ')
  #compute number of iterations per dataset epoch
  nbIterationPerEpoch=nbSamples/(usersettings.batch_size)
  print('One {mod} epoch performed in {iterations} iterations'.format(mod=mode, iterations=nbIterationPerEpoch))
  if nbIterationPerEpoch==0:
    raise ValueError('usersettings.nb_{mod}_samples is too low v.s. batch_size, check those values'.format(mod=mode))

  return nbIterationPerEpoch

def getTrainSpecs(params, global_hooks):
  ''' setup the training specs wrt the experiment usersettings file
  Args:
    param: a dictionnary of general parameters
    global_hooks : a list of hooks that should be considered at least in the train mode
                   this list will be completed by train mode specific hooks
  Returns:
    an initialized tf.estimator.TrainSpec object instance
  '''
  #copy the global hooks before adding others specific to the train mode
  train_hooks=copy.copy(global_hooks)

  #set the train input data pipeline and related hooks (if any)
  with tf.variable_scope('train_input_pipeline'),tf.device('/cpu:0'):
    # Setup data loaders
    train_input_fn, train_input_hook = usersettings.get_input_pipeline_train_val(
                                            batch_size=usersettings.batch_size,
                                            raw_data_files_folder=usersettings.raw_data_dir_train,
                                            shuffle_batches=True)
    if train_input_hook is not None:
      train_hooks.append(train_input_hook)

  #add a step counter
  train_hooks.append(tf.train.StepCounterHook(
                                            every_n_steps=10,
                                            every_n_secs=None,
                                            output_dir=params.sessionFolder,
                                            summary_writer=None
                                            )
                    )

  return tf.estimator.TrainSpec(input_fn=train_input_fn,
                                max_steps=params.nbIterationPerEpoch_train*usersettings.nbEpoch,
                                hooks=train_hooks)

def getEvalSpecs(params, global_hooks):
  ''' setup the eval specs wrt the experiment usersettings file
  Args:
    param: a dictionnary of general parameters
    global_hooks : a list of hooks that should be considered at least in the validation mode
                   this list will be completed by val mode specific hooks
  Returns:
    an initialized tf.estimator.EvalSpec object instance
  '''
  #copy the global hooks before adding others specific to the validation mode
  eval_hooks=copy.copy(global_hooks)

  with tf.variable_scope('eval_input_pipeline'),tf.device('/cpu:0'):
    eval_input_fn, eval_input_hook = usersettings.get_input_pipeline_train_val(
                                            batch_size=usersettings.batch_size,
                                            raw_data_files_folder=usersettings.raw_data_dir_val,
                                            shuffle_batches=False)
    if eval_input_hook is not None:
      eval_hooks.append(eval_input_hook)

  #specify model exporters to generate served models after each val session
  exporters = []
  #-> an exporter for intermediate models along training
  exporters.append(tf.estimator.LatestExporter(
        name='latest_models',
        serving_input_receiver_fn=usersettings.get_input_pipeline_serving,
        assets_extra=None,
        as_text=False,
        exports_to_keep=5
        )
      )

  try: #when possible, keep the best model (available in tf 1.10):
      tf.estimator.BestExporter(#choosing here to only export better models
          name='best_model',
          serving_input_receiver_fn=usersettings.get_input_pipeline_serving,
          event_file_pattern='eval/*.tfevents.*',
          compare_fn=_loss_smaller,
          assets_extra=None,
          as_text=False,
          exports_to_keep=5
          )
  except:
     print('tf.estimator.BestExporter is not available on this tensorflow version, tf>1.1 required')
     pass
  return tf.estimator.EvalSpec( input_fn=eval_input_fn,
                                steps=params.nbIterationPerEpoch_val,
                                name=None,
                                hooks=eval_hooks,
                                exporters=exporters,
                                start_delay_secs=120,
                                throttle_secs=60)

def getSessionConfig(params):
  '''
  defines the session configuration (GPU, summary options, etc.)
  Args:
    params: general parameters dictionnary
  Returns:
    a configured tf.estimator.RunConfig object instance
  '''
  gpu_options=tf.GPUOptions(allow_growth=True)
  #activate XLA JIT level 1 by default
  graph_options=tf.GraphOptions()
  if hasattr(usersettings,'XLA_FLAG'):
    graph_options.optimizer_options.global_jit_level = usersettings.XLA_FLAG
  else:
    graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF#ON_1#OFF
  sessionConfig=tf.ConfigProto(
                              allow_soft_placement=True,
                              log_device_placement=params.debug_sess,
                              gpu_options=gpu_options,
                              graph_options=graph_options
                              )

  # Set the run_config and the directory to save the model and stats
  summary_steps_period=1 #by default, log each step
  if usersettings.nb_summary_per_train_epoch>0:
    summary_steps_period=int(params.nbIterationPerEpoch_train/usersettings.nb_summary_per_train_epoch)
  run_config =tf.estimator.RunConfig(
                              model_dir=params.sessionFolder,
                              tf_random_seed=usersettings.random_seed,
                              save_summary_steps=summary_steps_period,
                              save_checkpoints_steps=params.nbIterationPerEpoch_train,
                              save_checkpoints_secs=None,
                              session_config=sessionConfig,
                              keep_checkpoint_max=5,
                              keep_checkpoint_every_n_hours=12,
                              log_step_count_steps=100,
                              train_distribute=None,
                              #TODO, activate when tf 1.10 available : device_fn=None
                              )

  return run_config

# Define and run experiment ###############################
def run_experiment(argv=None):
  print('Running an experiment. argv='+str(argv))

  # Define model parameters
  params = tf.contrib.training.HParams(
    nbIterationPerEpoch_train=getIterationsPerEpoch('train'),
    nbIterationPerEpoch_val=getIterationsPerEpoch('val'),
    learning_rate=usersettings.initial_learning_rate,
    debug=usersettings.display_model_layers_info #can be forced to True if script option --debug is provided
    )
  #add additionnal hyperparams coming from argv
  if argv is not None:
    if  isinstance(argv[0], dict):
      for key, val in argv[0].iteritems():
        print('Adding hyperparameter (key,val):'+str((key,val)))
        params.add_hparam(name=key,value=val)

  #specify general hooks (common to train and validate modes)
  globalHooks=[]
  if params.debug_sess:
    globalHooks.append(tf_debug.TensorBoardDebugHook(params.debug_server_addresses,
                                          send_traceback_and_source_code=True,
                                          log_usage=False)
                      )

  #specify the training input function and related parameters
  train_spec = getTrainSpecs(params, globalHooks)

  #specify the testing input function and related parameters
  eval_spec = getEvalSpecs(params, globalHooks)

  #specify session hardware configuration :
  run_config =getSessionConfig(params)

  # Define the estimator
  estimator = None
  if hasattr(usersettings, 'premade_estimator'):
    print('Using a premade estimator')
    #raise ValueError('TODO')
    estimator=usersettings.premade_estimator
    #estimator.model_dir=params.sessionFolder
  else:
    print('Using a custom estimator')
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config
    )

  #start the train/val/export session
  tf.estimator.train_and_evaluate(
            estimator,
            train_spec,
            eval_spec
        )

# Define model ############################################
def model_fn(features, labels, mode, params):
    """Model function used in the estimator.

    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (tf.estimator.ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyperparameters.

    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """

    print('###################################################')
    print('Defining the custom model_fn with mode : '+str(mode))
    print('=> input features='+str(features))
    if isinstance(features,dict):
        #basic case (for serving especially) where input is a dict with only the 'feature' item
        if 'feature' in features and len(features)==1:
            print('Found features dictionnary with unique key \'feature\', using as is')
            features=features['feature']
        elif hasattr(usersettings, 'features_dict_to_tensor'):
            features=usersettings.features_dict_to_tensor(features)
        else:
            raise ValueError('input features tensor is a dict, then, settings file MUST implement function features_dict_to_tensor(features): returns dense tensor to convert dict to the appropriate format. Received features: '+str(features))
    print('features='+str(features))

    #FIXME for now tensorflow_server only works on CPU so using GPU only for training and validation
    model_placement="/cpu:0"
    if mode != tf.estimator.ModeKeys.PREDICT and len(usersettings.used_gpu_IDs)>0:
        model_placement="/gpu:0"
        print('**** model placed on GPU')
    else:
        print('**** model placed on CPU')

    with tf.name_scope("data_preprocess"):
        features=usersettings.data_preprocess(features, model_placement)
    #FIXME currently not able to put model on a GPU... variables saving issue
    model_scope='model'
    with tf.device(model_placement), tf.variable_scope(model_scope):
        model=loadModel(params.sessionFolder)
        model_outputs_dict=model(   data=features,
                                    hparams=params, #hyperparameters that may control model settings
                                    mode=mode
                                )

        print('==> Model specified in \"{modelFile}\" generates the following outputs:'.format(modelFile=usersettings.model_file))
        for key, value in model_outputs_dict.items():
            print('->'+str((key, value)))

    if usersettings.predict_using_smoothed_parameters is True:
        #TODO, have a look here to fix current issues : from https://medium.freecodecamp.org/how-to-deploy-an-object-detection-model-with-tensorflow-serving-d6436e65d1d9
        #A confident demo: https://cloud.google.com/tpu/docs/inception-v3-advanced#exponential_moving_average

        with tf.device("/cpu:0"),tf.variable_scope('moving_average_trainables_saver'), tf.device("/cpu:0"):

            #define the moving average operator and its variables to smooth
            ema = tf.train.ExponentialMovingAverage(
                decay=MOVING_AVERAGE_DECAY, num_updates=tf.train.get_global_step())
            variables_to_average = tf.trainable_variables()
            #define the smoothing op and add to UPDATE_OPS collection that are run after each training step
            maintain_averages_op = ema.apply(variables_to_average)
            tf.add_to_collection('WEIGHTS_EMA_UPDATE_OP', maintain_averages_op)

            if params.debug: #plot the first layer weights sum to compare the variable and the ema version
                tf.summary.scalar(trainables[0].name, tf.reduce_sum(trainables[0]))
                tf.summary.scalar(ema.average(trainables[0]).name, tf.reduce_sum(ema.average(trainables[0])))

            #add maintain_averages_op to collection tf.GraphKeys.UPDATE_OPS to force running before the optimization step
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, maintain_averages_op)

            '''FIXME : for now, this is the only way i found to force the use of the smoothed weights for prediction
               smmothed weights are loaded for each prediction call... ugly not ?
               A better way would be to replace the weights by their smoothed version when writing the served model
            '''
            if mode == tf.estimator.ModeKeys.PREDICT : #restore moving averaged variables to predict
                print('*** Adding an op to load smoother weights before prediction ***')
                def _restore_vars(ema):
                    ema_variables = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
                    return tf.group(*[tf.assign(x, ema.average(x)) for x in ema_variables])

                #force the model to get the restored variables before running
                with tf.control_dependencies([_restore_vars(ema)]):
                    features = tf.identity(features)


    # Loss, training and eval operations are not needed during inference.
    loss = None
    train_op = None
    evaluation_hooks = None
    eval_metric_ops = {}
    train_parameters_scaffold=None
    embedding_checkpoint_saver=None
    if mode != tf.estimator.ModeKeys.PREDICT: #if training or validation, but not predicting/serving, compute a loss, etc.
        with tf.name_scope('model_loss'):
          with tf.name_scope('regularization_loss'):
            # -> first get the weights loss found in collection tf.GraphKeys.REGULARIZATION_LOSSES
            regularization_losses=tf.losses.get_regularization_losses()
            # list all weights
            if params.debug is True:
                print('Found the following regularisation losses')
                for layer_loss in regularization_losses:
                    print(layer_loss)
            print('Found {nb_losses} layers regularisation_losses within collection tf.GraphKeys.REGULARIZATION_LOSSES'.format(nb_losses=len(regularization_losses)))
            weights_loss=tf.reduce_sum(regularization_losses)#tf.losses.get_regularization_loss()
            tf.summary.scalar('Regularization_loss', weights_loss)
          #finalize total loss
          loss=usersettings.get_total_loss(inputs=features, model_outputs_dict=model_outputs_dict, labels=labels, weights_loss=weights_loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            '''define the training op that will first apply all ops found in collection
            tf.GraphKeys.UPDATE_OPS (batchnorm updates and weights ema for instance)
            and then apply the optimization op
            '''
            train_op = get_train_op_fn(loss, params)
            number_of_parameters=0
            trainables=tf.trainable_variables()
            for var in trainables:
                trainable_nb_values=np.prod(var.get_shape().as_list())
                if params.debug:
                    print('trained variable with {nb} parameters : {tensor}'.format(nb=trainable_nb_values, tensor=var))
                if trainable_nb_values>1:
                    tf.summary.histogram(var.op.name, var)
                number_of_parameters+=trainable_nb_values
                #summaries.append(tf.summarize_activation(var.op.name, var))
            print('### Number of parameters : '+str(number_of_parameters))


        if mode==tf.estimator.ModeKeys.EVAL:
            #DATA EMBEDDING SECTION for the validation stap only
            with tf.device(':/cpu:0'),tf.variable_scope('evaluate'):
                #->
                '''flatten raw and code samples
                  -> check if one have one or many labels per input sample
                  -->  sample level classification (as for image classification) : one label per sample, all validation dataset stored !!!
                  -->  multiple labels per sample (as for image semantic classification): many samples(ex:pixels) per data sample, only saving a fraction of them
                '''

                #flatten central data batches but keep the last dimension
                def get_flatten_feature(feature, feature_name):
                    ''' reshape each sample of the data batch to a simple vector
                    Args:
                      feature: the feature map to reshape
                      feature_name: the feature name to set a name to the reshape op
                    Returns: the flatten version
                    '''
                    print('Flattening feature map \'{name}\' : {tensor}'.format(name=feature_name, tensor=feature))
                    inputFeature_shape=feature.get_shape().as_list()
                    feature_shape=feature.get_shape().as_list()
                    if len(feature_shape)>1:
                        flatten_feature=tf.layers.flatten(feature)#reshape(feature, flatten_batch_feature_shape, name='flatten_'+feature_name)
                        print('---> flatten version : '+str(flatten_feature))
                    else:
                        if len(feature_shape)==0:
                            raise ValueError('This feature is a scalar, flattening does not make sense')
                        else:
                            print('---> already a flat tensor, returning as is, (shape,length)='+str((feature_shape, len(feature_shape))))
                            flatten_feature=feature
                    return flatten_feature

                xdimensions=len(features.get_shape().as_list())-2
                xdimensions_labels=len(labels.get_shape().as_list())-2
                print('xdimensions '+str(xdimensions)+' vs xdimensions_labels '+str(xdimensions_labels))
                #FIXME, the following criteria is stil hazardous and may nt adapt to new use cases
                denseLabels= (xdimensions>=2 and xdimensions_labels>=2) and len(features.get_shape().as_list())>=4
                if denseLabels is True: #multiple samples/labels per data sample use case
                    print('*** Dense labels case study')
                    print("Now preparing data embedding from the central pixels of the validation data samples")
                    #crop raw data as for labels whatever the dimension of the data (considering initial shape [batch, [xdimensions], channels])
                    xdimensions=(len(features.get_shape().as_list())-2)
                    def get_feature_central_area(feature_map, feature_name):
                        ''' returns a slice of the input feature map without border of size usersettings.field_of_view
                          Args:
                              feature_map: the input feature map
                          Returns:
                              the central part of the input feature but
                              keep as is if field of view is too large
                        '''
                        print('--> Extracting central patch AREA of feature map \'{name}\' : {tensor}'.format(name=feature_name, tensor=feature_map))
                        central_value=None
                        if usersettings.patchSize-usersettings.field_of_view>0:
                            additionnal_dims_size=features.get_shape().as_list()[3:-1]#remove spatial border effects but keep all the other dimensios as is
                            central_value=tf.slice( feature_map,
                                                  begin=[0]+[usersettings.field_of_view//2]*2+[0]*len(additionnal_dims_size)+[0],
                                                  size=[-1]+[usersettings.patchSize-usersettings.field_of_view]*2+additionnal_dims_size+[-1])
                        else:
                            central_value=feature_map
                        print('---> central value shape='+str(central_value.get_shape().as_list()))
                        return central_value
                    features_fov=get_feature_central_area(features, 'input')
                    model_outputs_fov={}
                    for output_key,output_feature in model_outputs_dict.items():
                      #check if spatial size matches between input and feature map, if yes, keep central area
                      if output_feature.get_shape().as_list()[1:3] == features.get_shape().as_list()[1:3]:
                        model_outputs_fov[output_key]= get_feature_central_area(output_feature, output_key)
                      else:#otherwise keep the data as is
                        model_outputs_fov[output_key]=output_feature
                    #pick the central pixel Value
                    def get_feature_central_pixel(feature_map, feature_name):
                        ''' returns the central pixel of the input feature map
                          Args:
                              feature_map: the input feature map
                          Returns:
                              the central pixel of the input feature
                        '''
                        print('--> Extracting central patch VALUE of feature map \'{name}\' : {tensor}'.format(name=feature_name, tensor=feature_map))

                        #get center coordinates
                        central_data_idx=(np.array(feature_map.get_shape().as_list()[1:3])/2).tolist()
                        print('---> central patch coordinates='+str(central_data_idx))
                        central_data_dims=len(features.get_shape().as_list()[3:])

                        #return the central slice
                        central_value= tf.slice( feature_map,
                                      begin=[0]+central_data_idx+[0]*central_data_dims,
                                      size=[-1]+[1,1]+[-1]*central_data_dims)
                        print('---> central patch VALUE shape='+str(central_value))
                        return central_value

                    model_outputs_center_val_dict={}
                    for output_key,output_feature in model_outputs_dict.items():
                      #FIXME test may not be robust enough...
                      if len(output_feature.get_shape().as_list())-2 == xdimensions:
                        model_outputs_center_val_dict[output_key]= get_feature_central_pixel(output_feature, output_key)
                      else:#keep the data as is
                        print('Could not extract central pixel of feature {feat}'.format(feat=output_feature))
                        model_outputs_center_val_dict[output_key]=output_feature
                    print('...Central pixel extraction OK')
                    labels_center_val=get_feature_central_pixel(labels, 'labels')
                    features_center_val=get_feature_central_pixel(features_fov, 'input')
                    '''#resize labels map to the size of the code to pick a rough label value consistent with the code
                    labels_resized_to_code_size=tf.image.resize_nearest_neighbor(
                      images=labels,
                      size=tf.constant(code_fov.get_shape().as_list()[1:(1+xdimensions)]),
                      align_corners=True,
                      name='labels_resizes_to_code_shape'
                    )
                    code_central_sample_label=tf.slice( labels_resized_to_code_size,
                                              begin=[0]+central_code_idx+[0],
                                              size=[-1]+[1]*xdimensions+[-1])
                    labels_center_val=tf.concat([labels_center_val,code_central_sample_label], axis=3)
                    '''
                    ''' selecting samples to store for embedding, keep only the center values
                      and try to keep connections between codes and labels before storing
                    '''
                    #deduce the maximum number of samples to store
                    stored_embedding_samples=params.nbIterationPerEpoch_val*usersettings.batch_size
                    flatten_features=get_flatten_feature(features_center_val, 'input_features')
                    flatten_labels=get_flatten_feature(labels_center_val, 'labels')
                    flatten_saved_samples_dict={}
                    for key, output_center in model_outputs_center_val_dict.items():
                        try:
                            flat_feature=get_flatten_feature(output_center, key)
                            flatten_saved_samples_dict[key]=flat_feature
                        except :
                            print('Scalar feature not considered for embedding projection')
                    eval_metric_ops = usersettings.get_eval_metric_ops(inputs=features_fov, model_outputs_dict=model_outputs_fov, labels=labels)

                else: #sample level classification
                    print('*** No dense labels case study')
                    stored_embedding_samples=params.nbIterationPerEpoch_val*usersettings.batch_size
                    flatten_features=get_flatten_feature(features, 'input_features')
                    flatten_labels=get_flatten_feature(labels, 'labels')
                    flatten_saved_samples_dict={}
                    for key, output_fov in model_outputs_dict.items():
                        try:
                            flat_feature=get_flatten_feature(output_fov, key)
                            flatten_saved_samples_dict[key]=flat_feature
                        except :
                            print('Scalar feature not considered for embedding projection')
                    eval_metric_ops = usersettings.get_eval_metric_ops(inputs=features, model_outputs_dict=model_outputs_dict, labels=labels)

                with tf.variable_scope('save_embeddings'):

                    #add input and label samples to store to the flatten_saved_samples dictionnary
                    flatten_saved_samples_dict['input_samples']=flatten_features
                    flatten_saved_samples_dict['labels']=flatten_labels
                    print('About to save, each iteration, the following data:'+str(flatten_saved_samples_dict))

                    '''-> prepare large buffers to store all evaluation samples for plotting
                      --> those buffer are LOCAL_VARIABLES dedicated to the EVAL mode
                      and not saved by tf.train.Saver of the TRAIN mode'''
                    def create_sample_values_pipeline_saver(features_to_save,
                                                            stored_embedding_samples,
                                                            name):
                        ''' prepare all the variable, tools and related ops to save data samples
                            for embedding visualization
                        '''
                        print('***Creating save embedding pipeline for variable \'{name}\': {tensor}'.format(name=name, tensor=features_to_save))
                        #define a queue dedicated to those samples saving
                        embedding_queue_capacities=stored_embedding_samples+usersettings.batch_size
                        samples_saving_queue=tf.FIFOQueue(capacity=embedding_queue_capacities,
                                                   dtypes=features_to_save.dtype.name,#'float',
                                                   shapes=features_to_save[0].get_shape(),
                                                   name=name+'_samples_queue')

                        #-> define the enqueing op
                        samples_enqueue=samples_saving_queue.enqueue_many(features_to_save)

                        if name == 'labels':
                            return {'queue':samples_saving_queue,
                                    'enqueue_op':samples_enqueue}
                        #create the buffer to save for the embeddings projector on the TensorBoard
                        whole_samples_to_store=tf.Variable(tf.zeros([stored_embedding_samples,features_to_save.get_shape().as_list()[-1]],dtype=features_to_save.dtype.name),
                                                                        trainable=False,
                                                                        collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                                                        name=name)
                        #-> define the final assign op that dequeues all the sample and store into the buffer
                        assign_samples=whole_samples_to_store.assign(samples_saving_queue.dequeue_up_to(stored_embedding_samples))#flatten_raw_images)
                        #assign_samples=tf.Print(assign_samples,[samples_saving_queue.size()], message="###################################################### Samples queue AFTER dequeue")#-> define a histogram on this buffer for monitoring purpose
                        #assign_samples=tf.Print(assign_samples,[assign_samples, samples_saving_queue.size()], message="###################################################### Samples queue AFTER dequeue")#-> define a histogram on this buffer for monitoring purpose
                        samples_hist=tf.summary.histogram(name+'_values',whole_samples_to_store)

                        return {'variable_buffer':whole_samples_to_store,
                                'embedding_histogram':samples_hist,
                                'queue':samples_saving_queue,
                                'enqueue_op':samples_enqueue,
                                'assign_op':assign_samples}

                    #create the list of buffers to store and their tools
                    feed_embedding_op =[]
                    save_embeddings_op = []
                    save_embedding_histograms_op=[]
                    variables_embeddings_to_save=[]
                    saved_variables_and_tools=[]
                    labels_queue=None
                    for data_key, data_to_save in flatten_saved_samples_dict.items():
                        print('preparing tools to save variable '+key)
                        single_data_and_tools=create_sample_values_pipeline_saver(
                                                                                features_to_save=data_to_save,
                                                                                stored_embedding_samples=stored_embedding_samples,
                                                                                name=data_key)

                        print(single_data_and_tools)
                        saved_variables_and_tools.append(single_data_and_tools)
                        feed_embedding_op.append(single_data_and_tools['enqueue_op'])
                        if data_key == 'labels': #labels are not stored but rather writen to a specific tsv format file
                            labels_queue=single_data_and_tools['queue']
                        else:
                            save_embedding_histograms_op.append(single_data_and_tools['embedding_histogram'])
                            variables_embeddings_to_save.append(single_data_and_tools['variable_buffer'])
                            save_embeddings_op.append(single_data_and_tools['assign_op'])

                    #define the saver that will write embedding variables to disk
                    embedding_checkpoint_saver=tf.train.Saver(variables_embeddings_to_save)
                    #force embeddings queue feeding at each step
                    with tf.control_dependencies(feed_embedding_op):
                        loss=tf.identity(
                                    loss,
                                    name='loss_eval_force_embedding_storing'
                                    )
                    '''Create an embedding projector configuration file (this tensor to associate to this metadata)
                       Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
                    '''
                    #create a metadata file to store the ground truth labels in
                    with tf.variable_scope('labels_to_metadata_file'):
                        labels_tsv_input=labels_queue.dequeue_up_to(stored_embedding_samples)
                        label_names=usersettings.reference_labels#label_names=['pixel_labels', 'code_labels'] #default labels that should be overiden by usersettings.reference_labels
                        print('declared labels in settings file='+str(label_names))
                        print('labels_tsv_input just dequeud:'+str(labels_tsv_input))

                        if flatten_labels.dtype.name != 'string':
                            labels_tsv_input=tf.as_string(labels_tsv_input)
                        print('labels_tsv_input just dequeud (string):'+str(labels_tsv_input))
                        if len(flatten_labels.get_shape().as_list())>1:
                            if flatten_labels.get_shape().as_list()[1]==1: #reduce to rank 0 if vector shape
                                labels_tsv_input=tf.reshape(labels_tsv_input,[-1])
                            else: #multiple labels case, add a title top line, separate coluns by tab ('\t')
                                print('Many labels available for labels tensor : '+str(labels_tsv_input))
                                if len(label_names)!=flatten_labels.get_shape().as_list()[1]:
                                    raise ValueError('experimentsSettingsFile.reference_labels label names list len({setupLen}) does not match labels len({pgmLen}), check your settings file'.format(setupLen=len(label_names), pgmLen=flatten_labels.get_shape().as_list()[1]))
                                labels_tsv_input=tf.concat([tf.constant([label_names]), labels_tsv_input], axis=0)
                                print('labels_tsv_input_withLabels:'+str(labels_tsv_input))
                                labels_tsv_input=tf.reduce_join(
                                                            inputs=labels_tsv_input,
                                                            axis=1,
                                                            keep_dims=False,
                                                            separator='\t',
                                                            name='labels_table_to_TSV_format_as_single_string',
                                                            reduction_indices=None
                                                        )
                                print('labels_tsv_input_line_joined_plus_added_separator:'+str(labels_tsv_input))
                        print('Labels to write tensor='+str(labels_tsv_input))
                        tsv_format_labels=tf.reduce_join(
                                                        inputs=labels_tsv_input,
                                                        axis=0,
                                                        keep_dims=False,
                                                        separator='\n',
                                                        name='labels_table_to_TSV_format_as_single_string',
                                                        reduction_indices=None
                                                    )
                        metadata_path='metadata.tsv'
                        metadata_path_abs=os.path.join(params.sessionFolder,embeddingsFolder, metadata_path)
                        write_medatata_file=tf.write_file(
                                                    filename=metadata_path_abs,
                                                    contents=tsv_format_labels,
                                                    name='write_medatata_file'
                                                    )
                        save_embeddings_op.append(write_medatata_file)
                        from tensorflow.contrib.tensorboard.plugins import projector
                        embeddings_summary_writer = tf.summary.FileWriter(os.path.join(params.sessionFolder,embeddingsFolder))
                        config = projector.ProjectorConfig()

                        # add multiple embeddings.
                        for saved_variable in variables_embeddings_to_save:
                            embedding_data_to_embed = config.embeddings.add()
                            embedding_data_to_embed.tensor_name = saved_variable.name
                            # Link this tensor to its metadata file (e.g. labels).
                            embedding_data_to_embed.metadata_path = metadata_path
                        # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
                        # read this file during startup.
                        writeMetadata_op=projector.visualize_embeddings(embeddings_summary_writer, config)

                    class FinalOpsHook(tf.train.FinalOpsHook):
                        def __init__(self, final_ops,final_ops_feed_dict=None, saver=None, summary_writer=None):
                            """Initializes `FinalOpHook` with ops to run and write at the end of the session.
                            Args:
                              final_ops: A dict of lists of ops separated by names 'final_ops' for standard ops and 'hist_ops' for histograms runs.
                              summary_writer : A writer `FileWriter` that will write the ops
                              final_ops_feed_dict: A feed dictionary to use when running `final_ops_dict`.
                              step : The step at which your ops are evaluated in the training program
                            """
                            self._final_ops = final_ops['final_ops']
                            self._vars=final_ops['vars']
                            self._hist_ops = final_ops['hist_ops']
                            self._final_ops_feed_dict = final_ops_feed_dict
                            self._saver=saver
                            self._summary_writer=summary_writer

                        def end(self, session):
                            print('**** FINALIZING EVAL SESSION...')
                            if self._vars is not None:
                                print('Saving all model parameters to pandas file...')
                                print('vars='+str(self._vars))
                                model_variables=pd.DataFrame({var.name:[session.run(var)] for var in self._vars})
                                model_variables.to_pickle(os.path.join(params.sessionFolder,'model_parameters.bz2'))
                                print('==>Values='+str(model_variables))

                            if self._final_ops is not None:
                                print('Saving embeddings and summaries')
                                print('This step may LOCK if nb_test_samples if larger than one epoch on the validation dataset')
                                for op in self._final_ops:
                                    print('Running '+str(op))
                                    session.run(op)
                                if self._summary_writer is not None:
                                    print('Writing summaries')
                                    for op in self._hist_ops:
                                        print('Running '+str(op))
                                        result=session.run(op)
                                        self._summary_writer.add_summary(result)
                                    #force summary write to file
                                    self._summary_writer.flush()
                                print('Saving embeddings in folder : '+str(embeddingsFolder))
                                self._saver.save(session,os.path.join(params.sessionFolder,embeddingsFolder,'embedding_values'))
                                print('**** EVAL SESSION FINISHED ****')

                #get all model variables
                all_variables_states=None
                if hasattr(usersettings, 'save_model_variables_to_pandas'):
                    if usersettings.save_model_variables_to_pandas is True:
                        all_variables_states=tf.global_variables(scope=model_scope)

                eval_finalize_hook=FinalOpsHook(final_ops={'final_ops':save_embeddings_op,'vars':all_variables_states, 'hist_ops':save_embedding_histograms_op},
                                 final_ops_feed_dict=None,
                                 saver=embedding_checkpoint_saver,
                                 summary_writer=embeddings_summary_writer)
                evaluation_hooks=[eval_finalize_hook]

                if hasattr(usersettings, 'get_validation_summaries'):
                  with tf.name_scope('eval_addon_summaries'):
                    if denseLabels is True:
                      eval_addon_summaries, save_steps=usersettings.get_validation_summaries(inputs=features_fov, model_outputs_dict=model_outputs_fov, labels=labels)
                    else:
                      eval_addon_summaries, save_steps=usersettings.get_validation_summaries(inputs=features, model_outputs_dict=model_outputs_dict, labels=labels)
                  eval_summary_hook = tf.train.SummarySaverHook(
                                save_steps=save_steps,
                                output_dir= os.path.join(params.sessionFolder,embeddingsFolder,'eval_addon_summaries'),
                                summary_op=tf.summary.merge(eval_addon_summaries, 'eval_addon_summaries'))
                  # Add it to the evaluation_hook list
                  evaluation_hooks.append(eval_summary_hook)


                # smoothed parameters load eval hook
                class LoadEMAHook(tf.train.SessionRunHook):
                  def __init__(self, model_dir):
                    super(LoadEMAHook, self).__init__()
                    self._model_dir = model_dir

                  def begin(self):
                    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
                    variables_to_restore = ema.variables_to_restore()
                    print('Variables to restore:')
                    print(variables_to_restore)
                    self._load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
                        tf.train.latest_checkpoint(self._model_dir),
                        variables_to_restore,
                        ignore_missing_vars=True)

                  def after_create_session(self, sess, coord):
                    tf.logging.info('********** Reloading moving averaged parameters ************')
                    self._load_ema(sess)
                    tf.logging.info('Done')

                if usersettings.predict_using_smoothed_parameters is True:
                    evaluation_hooks.append(LoadEMAHook(params.sessionFolder))
                    #ema_variables = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
                    #return tf.group(*[tf.assign(x, ema.average(x)) for x in ema_variables])


    with tf.name_scope('model_outputs_postprocessing'):
        exported_outputs=usersettings.model_outputs_postprocessing_for_serving(model_outputs_dict)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=exported_outputs,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs={key:tf.estimator.export.PredictOutput({key:output}) for key,output in exported_outputs.items()},
        evaluation_hooks=evaluation_hooks
    )

def get_train_op_fn(loss, params):
    """Get the training Op.

    Args:
         loss (Tensor): Scalar Tensor that represents the loss function.
         params (HParams): Hyperparameters (needs to have `learning_rate`)

    Returns:
        Training Op
    """
    global_step = tf.train.get_global_step()#tf.train.get_or_create_global_step()

    print('Creating solver...')
    with tf.name_scope('optimizer'):
        if usersettings.num_epochs_per_decay>0 and usersettings.learning_rate_decay_factor>0:
            #TODO: check for cold epoch/warmup/exponential learning rate profiles : https://cloud.google.com/tpu/docs/inception-v3-advanced#exponential_moving_average
            with tf.name_scope('learning_rate_decay'):
                # Calculate the learning rate schedule.
                decay_steps = int(params.nbIterationPerEpoch_train * usersettings.num_epochs_per_decay)

                # Decay the learning rate exponentially based on the number of steps.
                lr = tf.train.exponential_decay(params.learning_rate,
                                              global_step,
                                              decay_steps,
                                              usersettings.learning_rate_decay_factor,
                                              staircase=True)
                tf.summary.scalar('learning_rate', lr)
        else:
            lr=params.learning_rate

        #get all extra ops to be ran before optimisation (including batch norm updates)
        model_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #define the optimizer, no more forcing to be on the CPU side: (do not force on specific device since backprop, etc should be done where forwrd pass is done
        with tf.control_dependencies(model_extra_update_ops):
          optimizer = usersettings.getOptimizer(loss=loss, learning_rate=lr, global_step=global_step)
        #update weights moving averages is expected to after the parameters update
        if usersettings.predict_using_smoothed_parameters is True:
          with tf.control_dependencies([optimizer]):
            optimizer = tf.group(*tf.get_collection('WEIGHTS_EMA_UPDATE_OP'))

    return optimizer

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
        print 'Server is ready'
        return True
      else:
        print('Error:'+str(error.details))
    return False
    time.sleep(1)


def _create_rpc_callback():
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
    exception = result_future.exception()
    if exception:
      #result_counter.inc_error()
      print(exception)
    else:
      try:
          if FLAGS.debug:
              print(result_future.result())
          response=usersettings.received_prediction_serving(result_future)
      except Exception,e:
          raise ValueError('Exception encountered on client callback : '.format(error=e))
  return _callback

def do_inference(host, port, model_name, concurrency, num_tests):
  """Tests PredictionService with concurrent requests.
  Args:
    host:tensorfow server address
    port: port address of the PredictionService.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use, infinite prediction loop if <0.
  Raises:
    IOError: An error occurred processing test data set.
  """
  from grpc.beta import implementations
  from grpc.framework.interfaces.face import face
  from tensorflow_serving.apis import predict_pb2
  from tensorflow_serving.apis import prediction_service_pb2


  print('Trying to interract with server:{srv} on port {port} for prediction...'.format(srv=host,
                                                         port=port))
  '''channels created from implementations.insecure_channel for now does not suppport large messages, following https://github.com/grpc/grpc/issues/13497
  #-> then, the bellow function overrides to solve the problem
  channel = implementations.insecure_channel(host, int(port))
  FIXME : to be updated when libraries get more stable
  '''
  import grpc.beta.implementations
  from grpc._cython import cygrpc

  def insecure_channel(host, port):
        channel = grpc.insecure_channel(
            target=host if port is None else '%s:%d' % (host, port),
            options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                     (cygrpc.ChannelArgKey.max_receive_message_length, -1)])
        return grpc.beta.implementations.Channel(channel)
  channel = insecure_channel(host, int(port))
  #channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  #allocate a clientIO instance defined for the experiment
  client_io=usersettings.Client_IO(FLAGS.debug)
  notDone=True
  predictionIdx=0
  while notDone:
      try:
        predictionIdx=predictionIdx+1
        start_time=time.time()
        sample=client_io.getInputData(predictionIdx)
        if FLAGS.debug:
            print('Input data is ready (data, shape)'+str((sample, sample.shape)))
            print('Time to prepare collect data request:',round(time.time() - start_time, 2))
            start_time=time.time()
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.signature_name = usersettings.served_head
        request.inputs[usersettings.input_data_name].CopyFrom(
              tf.make_tensor_proto(sample, shape=sample.shape))
        if FLAGS.debug:
          print('Time to prepare request:',round(time.time() - start_time, 2))
      except StopIteration:
        print('End of the process detection, finalizing the finalize method')
        notDone=True
        break
      #asynchronous message reception, may hide some AbortionError details and only provide CancellationError(code=StatusCode.CANCELLED, details="Cancelled")
      '''result_future = stub.Predict.future(request, usersettings.serving_client_timeout_int_secs)  # 5 seconds
      result_future.add_done_callback(
            _create_rpc_callback())
      '''
      #synchronous approach... that may provide more details on AbortionError
      if FLAGS.debug:
          print(stub.Predict(request, usersettings.serving_client_timeout_int_secs))
          start_time=time.time()
      answer=stub.Predict(request, usersettings.serving_client_timeout_int_secs)
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
        filename=os.path.join(restart_from_sessionFolder, settingsFile_saveName)
        print('From working folder'+str(os.getcwd()))
        print('looking for '+str(filename))
        if os.path.exists(filename):
          print('Found')
        else:
          raise ValueError('Could not find experiment_settings.py file in the experiment folder:'+str(sessionFolder))
      else:
        raise ValueError('Could not restart interrupted training session, working folder not found:'+str(model_dir))
    else:
      print('Process starts...')

    print('Trying to load experiments settings file : '+str(filename))
    try:
        usersettings=imp.load_source('settings', filename)
    except Exception,e:
        raise ValueError('Failed to load {settings} file : '.format(settings=filename, error=e))
    print('loaded settings file {file}'.format(file=filename))

    settings_checker=ExperimentsSettingsChecker(usersettings)
    settings_checker.validate_settings(isServingModel)

    if len(usersettings.used_gpu_IDs)>=1:
        print('Forcing system to only focus on the target GPU {gpuID} thus avoiding memory allocation issues on the other GPUs'.format(gpuID=usersettings.used_gpu_IDs))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(usersettings.used_gpu_IDs)[1:-1]

    if hasattr(usersettings, 'model_file'):
      model_name=usersettings.model_file.split('.')[0]
    else:
      model_name='premade_estimator'
    #manage the working folder in the case of a new experiment
    workingFolder=usersettings.workingFolder
    if restart_from_sessionFolder is None:
      sessionFolder=os.path.join(workingFolder, usersettings.session_name+'_'+datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
    return usersettings, sessionFolder, model_name

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
  served_model_info_cmd='saved_model_cli show --dir {target_model} --tag_set serve'.format(target_model=one_model_path)
  print('Checking served model available signatures using command '+served_model_info_cmd)
  cmd_result=subprocess.check_output(served_model_info_cmd.split())
  print('You may add option \' --signature_def SIGNATURE_DEF_NAME\' to get details on inputs and outputs of the model')
  print('Answer='+str(cmd_result))
  if expected_model_name in cmd_result:
    print('Target model {target} name found in the command answer'.format(target=expected_model_name))
  else:
    raise ValueError('Target model {target} name NOT found in the command answer'.format(target=expected_model_name))

# Run script ##############################################
if __name__ == "__main__":
    ''' main function that starts the experiment in the chosen mode '''
    scripts_WD=os.getcwd() #to locate the mysettings*.py file

    if FLAGS.debug is True:
        raw_input('Running in debug mode. Press Enter to continue...')
    if FLAGS.start_server is True:
        print('### START TENSORFLOW SERVER MODE ###')

        usersettings, sessionFolder, model_name = loadExperimentsSettings(os.path.join(scripts_WD,FLAGS.model_dir,settingsFile_saveName), isServingModel=True)

        #target the served models folder
        model_folder=os.path.join(scripts_WD,FLAGS.model_dir,'export/best_model')
        if not(os.path.exists(model_folder)):
          model_folder=os.path.join(scripts_WD,FLAGS.model_dir,'export/latest_models')
        print('Considering served model parent directory:'+model_folder)
        #check if at least one served model exists in the target models directory
        stillWait=True
        while stillWait is True:
          print('Looking for a servable model in '+os.path.join(scripts_WD,FLAGS.model_dir,'export/'))
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
            # print servable informations
            #propose some commands to get information on the served model
            print('If necessary, check the served model behaviors using command line cli : saved_model_cli show --dir path/to/export/model/latest_model/1534610225/ --tag_set serve to get the MODEL_NAME(S)\n to get more details on the target MODEL_NAME, you can then add option --signature_def MODEL_NAME')
          except Exception, e:
            raise ValueError('Could not find servable model, error='+str(e.message))

        get_served_model_info(one_model_path, usersettings.served_head)
        tensorflow_start_cmd="tensorflow_model_server --port={port} --model_name={model} --model_base_path={model_dir}".format(port=usersettings.tensorflow_server_port,
                                                                                                                model=model_name,
                                                                                                                model_dir=model_folder)

        print('Starting tensorflow server with command :'+tensorflow_start_cmd)
        os.system(tensorflow_start_cmd)

    elif FLAGS.predict is True or FLAGS.predict_stream !=0:
        print('### PREDICT MODE, interacting with a tensorflow server ###')
        print('If necessary, check the served model behaviors using command line cli : saved_model_cli show --dir path/to/export/model/latest_model/1534610225/ --tag_set serve to get the MODEL_NAME(S)\n to get more details on the target MODEL_NAME, you can then add option --signature_def MODEL_NAME')

        usersettings, sessionFolder, model_name = loadExperimentsSettings(os.path.join(scripts_WD,FLAGS.model_dir,settingsFile_saveName), isServingModel=True)

        #FIXME errors reported on gRPC: https://github.com/grpc/grpc/issues/13752 ... stay tuned, had to install a specific gpio version (pip install grpcio==1.7.3)
        server_ready=WaitForServerReady(usersettings.tensorflow_server_address, usersettings.tensorflow_server_port)
        if server_ready is False:
            raise ValueError('Could not reach tensorflow server')
        print('Prediction mode using model : '+FLAGS.model_dir)
        predictions_dir=os.path.join(FLAGS.model_dir,
                                'predictions_'+datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
        os.mkdir(predictions_dir)
        os.chdir(predictions_dir)
        print('Current working directory = '+os.getcwd())
        do_inference(usersettings.tensorflow_server_address, usersettings.tensorflow_server_port, model_name, 0, FLAGS.predict_stream)

    elif FLAGS.commands is True or FLAGS.commands is True:
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
        usersettings, sessionFolder, model_name = loadExperimentsSettings(FLAGS.usersettings, FLAGS.model_dir)

        argv_app={'debug_server_addresses':FLAGS.debug_server_addresses, 'sessionFolder':sessionFolder, 'model_name':model_name, 'debug_sess':FLAGS.debug}
        #add additionnal hyperparams coming from an optionnal
        if hasattr(usersettings, 'hparams'):
          print('adding hypermarameters declared from the experiments settings script')
          argv_app.update(usersettings.hparams)
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
            argv_app.update({'sessionFolder':sessionFolder})
        #copy settings and model file to the working folder
        if not FLAGS.restart_interrupted:
          os.makedirs(sessionFolder)
          os.makedirs(os.path.join(sessionFolder,embeddingsFolder))
          if hasattr(usersettings, 'model_file'):
            shutil.copyfile(os.path.join(scripts_WD, usersettings.model_file), os.path.join(sessionFolder, usersettings.model_file))
          settings_copy_fullpath=os.path.join(sessionFolder, settingsFile_saveName)
          shutil.copyfile(os.path.join(scripts_WD, FLAGS.usersettings), settings_copy_fullpath)



        tf.app.run(
            main=run_experiment,
            argv=[argv_app]
    )
