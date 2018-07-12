"""
#What's that ?
A set of script that demonstrate the use of Tensorflow experiments and estimators on 1D data
@brief : the main script that enables training, validation and serving Tensorflow based models merging all needs in a
single script to train, evaluate, export and serve.
taking large inspirations of official tensorflow demos.
@author : Alexandre Benoit, LISTIC lab, FRANCE

Several ideas are put together:
-experiments and estimators to manage training, valiation and export in a easier way (but experiments are still in the contrib module so subject to strong changes)
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
-moving average parameters saving is maybe not correctly done. I am not sure that the smoothed variables are saved instead of the current parameters
-for now tensorflow_server only works on CPU so using GPU only for training and validation. Track : https://github.com/tensorflow/serving/issues/668

#TODO :

To adapt to new use case, just update the mysettingsxxx file and adjust I/O functions.
For any experiment, the availability of all the required fields in the settings file is checked by the experiments_settings_checker.py script. You can have a look there to ensure you prepared everything right.

As a reminder, here are the functions prototypes:
-define a model to be trained and served in a specific file and follow this prototype:
--report model name in the settings file
--def model( data, #the input data tensor
            n_outputs, #dimension of the embedding code
            hparams,  #external parameters that may be used to setup the model
            mode), #mode set to switch between train, validate and inference mode
            wrt tensorflow.contrib.learn.ModeKeys values
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

Some examples of such functions are put in the REAME.md and in the versionned mysettings_xxx.py demos

This demo relies on Tensorflow 1.4.1 and makes use of the Experiment and Estimator
available in the contrib module that is expected to evolve along tensorflow versions.
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
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner

from tensorflow.python import debug as tf_debug

global usersettings
embeddingsFolder='embeddings'
settingsFile_saveName='experiment_settings.py'

MOVING_AVERAGE_DECAY=0.9999
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
        raise ValueError('Failed to load model file : '+str(usersettings.model_file)+str(e))
    model=model_def.model

    print('loaded model file {file}'.format(file=model_path))
    return model

# Define and run experiment ###############################
def run_experiment(argv=None):
    print('Running an experiment. argv='+str(argv))
    """Run the training experiment."""
    nbIterationPerEpoch_train=usersettings.nb_train_samples/(usersettings.batch_size)#len(dataset_raw_train)*patchesPerImage/batch_size
    nbIterationPerEpoch_test=usersettings.nb_test_samples/(usersettings.batch_size)
    print('One TRAIN epoch performed in {iterations} iterations'.format(iterations=nbIterationPerEpoch_train))
    print('One TEST epoch performed in {iterations} iterations'.format(iterations=nbIterationPerEpoch_test))
    if nbIterationPerEpoch_train==0:
        raise ValueError('usersettings.nb_train_samples is too low v.s. batch_size, check those values')
    if nbIterationPerEpoch_test==0:
        raise ValueError('usersettings.nb_test_samples is too low v.s. batch_size, check those values')
    # Define model parameters
    params = tf.contrib.training.HParams(
        learning_rate=usersettings.initial_learning_rate,
        n_classes=usersettings.nb_classes,
        train_steps=usersettings.nbEpoch*nbIterationPerEpoch_train,
        min_eval_frequency=nbIterationPerEpoch_train,#test after each train epoch
        nbIterationPerEpoch_test=nbIterationPerEpoch_test,
        nbIterationPerEpoch_train=nbIterationPerEpoch_train,
        debug=usersettings.display_model_layers_info
        )
    #add additionnal hyperparams coming from argv
    if argv is not None:
      if  isinstance(argv[0], dict):
        for key, val in argv[0].iteritems():
          print('Adding hyperparameter (key,val):'+str((key,val)))
          params.add_hparam(name=key,value=val)


    #Session hardware configuration :
    gpu_options=tf.GPUOptions(allow_growth=True)
    #activate XLA JIT level 1 by default
    graph_options=tf.GraphOptions()
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
        summary_steps_period=int(nbIterationPerEpoch_train/usersettings.nb_summary_per_train_epoch)
    run_config = tf.contrib.learn.RunConfig(#FIXME : to be replaced once allowed by tf.estimator.RunConfig
                                session_config=sessionConfig,
                                save_summary_steps=summary_steps_period,
                                save_checkpoints_steps=nbIterationPerEpoch_train,
                                model_dir=params.sessionFolder
                                            )

    '''TODO: have a look at :
    https://www.tensorflow.org/api_docs/python/tf/contrib/learn/learn_runner/tune
    related discussions :
    --> https://github.com/tensorflow/tensorflow/issues/7868
    --> https://github.com/tensorflow/tensorflow/issues/16576
    --> a proposal: https://github.com/makoeppel/hyperParameterSearchTF/tree/master
    tuner = create_tuner(study_configuration, objective_key)
    learn_runner.tune(experiment_fn=_create_my_experiment, tuner)
    '''

    learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule=usersettings.train_val_schedule_strategy,
        hparams=params  # HParams
    )

def experiment_fn(run_config, params):
    """Create an experiment to train and evaluate the model.

    Args:
        run_config (RunConfig): Configuration for Estimator run.
        params (HParam): Hyperparameters

    Returns:
        (Experiment) Experiment for training the mnist model.
    """
    # You can change a subset of the run_config properties as
    #run_config = run_config.replace(save_checkpoints_steps=params.min_eval_frequency)

    print('Starting experiment with Hyper parameters:',str(params))

    # Define the mnist classifier
    estimator = get_estimator(run_config, params)
    with tf.variable_scope('train_val_input_pipeline'),tf.device('/cpu:0'):
        # Setup data loaders
        train_input_fn, train_input_hook = usersettings.get_input_pipeline_train_val(
                                                    batch_size=usersettings.batch_size,
                                                    raw_data_files_folder=usersettings.raw_data_dir_train,
                                                    shuffle_batches=True)
        eval_input_fn, eval_input_hook = usersettings.get_input_pipeline_train_val(
                                                    batch_size=usersettings.batch_size,
                                                    raw_data_files_folder=usersettings.raw_data_dir_val,
                                                    shuffle_batches=False)

    '''#create an early stop monnitor (stop training when test loss no more decreases for a long time)
    early_stop_monitor = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                            min_delta=0,
                                                            patience=0,
                                                            verbose=1,
                                                            mode='auto'
                                                        )
    '''
    steps_counter_hook=tf.train.StepCounterHook(
                                                every_n_steps=10,
                                                every_n_secs=None,
                                                output_dir=params.sessionFolder,
                                                summary_writer=None
                                                )

    # TODO for model export and serving, have a look here : https://www.tensorflow.org/programmers_guide/saved_model#requesting_predictions_from_a_local_server
    # TODO, look example here : https://stackoverflow.com/questions/44535119/example-of-tensorflow-contrib-learn-exportstrategy
    export_strategy=tf.contrib.learn.make_export_strategy(
        serving_input_fn=usersettings.get_input_pipeline_serving,
        default_output_alternative_key=None,
        assets_extra=None,
        as_text=False,
        exports_to_keep=5
    )
    train_monitors=[steps_counter_hook]
    eval_hooks=None
    if params.debug_sess:
      debughook=tf_debug.TensorBoardDebugHook(params.debug_server_addresses,
                                            send_traceback_and_source_code=False,
                                            log_usage=False)
      #add debug hook to train and eval monitors
      eval_hooks=[debughook]
      train_monitors.append(debughook)

    # Define the experiment
    if train_input_hook is not None:
        train_monitors+=[train_input_hook]
    if eval_input_hook is not None:
        eval_hooks=[eval_input_hook]
    #CLI debug hook hooks = [tf_debug.LocalCLIDebugHook()]

    experiment = tf.contrib.learn.Experiment(
                    estimator=estimator,
                    train_input_fn=train_input_fn,
                    eval_input_fn=eval_input_fn,
                    eval_metrics=None,
                    train_steps=params.train_steps,
                    eval_steps=params.nbIterationPerEpoch_test,
                    train_monitors=train_monitors,
                    eval_hooks=eval_hooks,
                    eval_delay_secs=120,
                    continuous_eval_throttle_secs=60,
                    min_eval_frequency=params.min_eval_frequency,
                    delay_workers_by_global_step=False,
                    export_strategies=export_strategy,
                    train_steps_per_iteration=params.min_eval_frequency,
                    checkpoint_and_export=True
                )
    return experiment

# Define model ############################################
def get_estimator(run_config, params):
    """Return the model as a Tensorflow Estimator object.

    Args:
         run_config (RunConfig): Configuration for Estimator run.
         params (HParams): hyperparameters.
    """
    return tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )

def model_fn(features, labels, mode, params):
    """Model function used in the estimator.

    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyperparameters.

    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    if usersettings.random_seed is not None:
      tf.set_random_seed(usersettings.random_seed)
    print('############ Received features='+str(type(features)))
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
    if mode != ModeKeys.INFER:
        model_placement="/gpu:0"

    with tf.name_scope("data_preprocess"):
        features=usersettings.data_preprocess(features, model_placement)
    #FIXME currently not able to put model on a GPU... variables saving issue
    model_scope='model'
    with tf.device(model_placement), tf.variable_scope(model_scope):
        model=loadModel(params.sessionFolder)
        model_outputs_dict=model(   data=features,
                                    n_outputs=usersettings.nb_classes,
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

            ema = tf.train.ExponentialMovingAverage(
                decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
            variables_to_average = (tf.trainable_variables() +
                                    tf.moving_average_variables())
            with tf.control_dependencies([train_op]), tf.name_scope('moving_average'):
              maintain_averages_op = ema.apply(variables_to_average)

            if params.debug: #plot the first layer weights sum to compare the variable and the ema version
                tf.summary.scalar(trainables[0].name, tf.reduce_sum(trainables[0]))
                tf.summary.scalar(ema.average(trainables[0]).name, tf.reduce_sum(ema.average(trainables[0])))

            #add maintain_averages_op to collection tf.GraphKeys.UPDATE_OPS to force running before the optimization step
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, maintain_averages_op)

            if mode != ModeKeys.TRAIN : #restore moving averaged variables to predict
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
    if mode != ModeKeys.INFER: #if training or validation, but not predicting/serving, compute a loss, etc.
        with tf.name_scope('model_loss'):
          with tf.name_scope('regularization_loss'):
            # -> first get the weights loss found in collection tf.GraphKeys.REGULARIZATION_LOSSES
            regularization_losses=tf.losses.get_regularization_losses()
            # list all weights
            if params.debug is True:
                print('Found the following regularisation losses')
                for layer_loss in regularization_losses:
                    print(layer_loss)
            print('Found {nb_losses} layers regularisation_losses'.format(nb_losses=len(regularization_losses)))
            weights_loss=tf.reduce_sum(regularization_losses)#tf.losses.get_regularization_loss()
            tf.summary.scalar('Regularization_loss', weights_loss)
          #finalize total loss
          loss=usersettings.get_total_loss(inputs=features, model_outputs_dict=model_outputs_dict, labels=labels, weights_loss=weights_loss)

        if mode == ModeKeys.TRAIN:
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


        if mode==ModeKeys.EVAL:
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
                            central_value=tf.slice( feature_map,
                                                  begin=[0]+[usersettings.field_of_view//2]*xdimensions+[0],
                                                  size=[-1]+[usersettings.patchSize-usersettings.field_of_view]*xdimensions+[-1])
                        else:
                            central_value=feature_map
                        print('---> central value shape='+str(central_value.get_shape().as_list()))
                        return central_value
                    features_fov=get_feature_central_area(features, 'input')
                    model_outputs_fov={output_key: get_feature_central_area(output_feature, output_key) for output_key,output_feature in model_outputs_dict.items()}
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
                        central_data_idx=(np.array(feature_map.get_shape().as_list()[1:1+xdimensions])/2).tolist()
                        print('---> central patch VALUE index='+str(central_data_idx))
                        #return the central slice
                        central_value= tf.slice( feature_map,
                                      begin=[0]+central_data_idx+[0],
                                      size=[-1]+[1]*xdimensions+[-1])
                        print('---> central patch VALUE shape='+str(central_value))
                        return central_value

                    model_outputs_center_val_dict={output_key: get_feature_central_pixel(output_feature, output_key) for output_key,output_feature in model_outputs_dict.items()}
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
                    stored_embedding_samples=params.nbIterationPerEpoch_test*usersettings.batch_size
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
                    stored_embedding_samples=params.nbIterationPerEpoch_test*usersettings.batch_size
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
                    ''' WHEN AVAILABLE? ADD IMAGE summaries
                       ---> https://github.com/tensorflow/tensorflow/issues/15332
                       ---> https://github.com/tensorflow/tensorflow/issues/14042
                       #add image summaries if inputs are images
                    val_summaries=usersettings.get_validation_summaries(inputs=features, predictions=predictions, labels=labels, embedding_code=embedding_code)
                    if val_summaries is not None:
                        print('new summaries added!!!')
                    '''

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
                        assign_samples=whole_samples_to_store.assign(samples_saving_queue.dequeue_many(stored_embedding_samples))#flatten_raw_images)
                        #-> define a histogram on this buffer for monitoring purpose
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
                        labels_tsv_input=labels_queue.dequeue_many(stored_embedding_samples)
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
                    eval_addon_summaries, save_steps=usersettings.get_validation_summaries(inputs=features_fov, model_outputs_dict=model_outputs_fov, labels=labels)
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
                    self._load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
                        tf.train.latest_checkpoint(self._model_dir), variables_to_restore)

                  def after_create_session(self, sess, coord):
                    tf.logging.info('********** Reloading EMA... ************')
                    self._load_ema(sess)
                  if usersettings.predict_using_smoothed_parameters is True:
                    evaluation_hooks=eval_finalize_hook+[LoadEMAHook(params.sessionFolder)]

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
            tf.contrib.util.make_tensor_proto(sample, shape=sample.shape))
      if FLAGS.debug:
        print('Time to prepare request:',round(time.time() - start_time, 2))

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


def loadExperimentsSettings(filename, restart_from_sessionFolder=None):
    ''' load experiments parameters from the mysettingsxxx.py script
        also mask GPUs to only use the ones specified in the settings file
      Args:
        filename: the settings file, if restarting an interrupted training session, you should target the experiments_settings.py copy available in the experiment folder to restart"
        restart_from_sessionFolder: [OPTIONNAL] set the  session folder of a previously interrupted training session to restart
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
      print('Starting a new experiment')

    print('Trying to load experiments settings file : '+str(filename))
    try:
        usersettings=imp.load_source('settings', filename)
    except Exception,e:
        raise ValueError('Failed to load {settings} file : '.format(settings=filename, error=e))
    print('loaded settings file {file}'.format(file=filename))

    settings_checker=ExperimentsSettingsChecker(usersettings)
    settings_checker.validate_settings()

    if len(usersettings.used_gpu_IDs)>=1:
        print('Forcing system to only focus on the target GPU {gpuID} thus avoiding memory allocation issues on the other GPUs'.format(gpuID=usersettings.used_gpu_IDs))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(usersettings.used_gpu_IDs)[1:-1]

    model_name=usersettings.model_file.split('.')[0]
    #manage the working folder in the case of a new experiment
    workingFolder=usersettings.workingFolder
    if restart_from_sessionFolder is None:
      sessionFolder=os.path.join(workingFolder, usersettings.session_name+'_'+datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
    return usersettings, sessionFolder, model_name

# Run script ##############################################
if __name__ == "__main__":
    ''' main function that starts the experiment in the chosen mode '''
    scripts_WD=os.getcwd() #to locate the mysettings*.py file

    if FLAGS.debug is True:
        raw_input('Running in debug mode. Press Enter to continue...')
    if FLAGS.start_server is True:
        print('### START TENSORFLOW SERVER MODE ###')
        usersettings, sessionFolder, model_name = loadExperimentsSettings(os.path.join(scripts_WD,FLAGS.model_dir,settingsFile_saveName))

        tensorflow_start_cmd="tensorflow_model_server --port={port} --model_name={model} --model_base_path={model_dir}".format(port=usersettings.tensorflow_server_port,
                                                                                                                model=model_name,
                                                                                                                model_dir=os.path.join(scripts_WD,FLAGS.model_dir,'export/Servo'))

        print('Starting tensorflow server with command :'+tensorflow_start_cmd)
        os.system(tensorflow_start_cmd)

    elif FLAGS.predict is True or FLAGS.predict_stream !=0:
        print('### PREDICT MODE, interacting with a tensorflow server ###')
        usersettings, sessionFolder, model_name = loadExperimentsSettings(os.path.join(scripts_WD,FLAGS.model_dir,settingsFile_saveName))

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
    else:
        print('### TRAINING MODE ###')
        usersettings, sessionFolder, model_name = loadExperimentsSettings(FLAGS.usersettings, FLAGS.model_dir)
        #copy settings and model file to the working folder
        if not FLAGS.restart_interrupted:
          os.makedirs(sessionFolder)
          os.makedirs(os.path.join(sessionFolder,embeddingsFolder))
          shutil.copyfile(os.path.join(scripts_WD, usersettings.model_file), os.path.join(sessionFolder, usersettings.model_file))
          settings_copy_fullpath=os.path.join(sessionFolder, settingsFile_saveName)
          shutil.copyfile(os.path.join(scripts_WD, FLAGS.usersettings), settings_copy_fullpath)
        tf.app.run(
            main=run_experiment,
            argv=[{'debug_server_addresses':FLAGS.debug_server_addresses, 'sessionFolder':sessionFolder, 'model_name':model_name, 'debug_sess':FLAGS.debug}]
    )
