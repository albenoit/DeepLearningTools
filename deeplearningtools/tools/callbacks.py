# ========================================
# FileName: callbacks.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: Define callbacks process
# for DeepLearningTools.
# =========================================

from deeplearningtools.helpers.distance_metrics.metrics import METRIC_SET
import os
import pandas as pd
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import numpy as np

class MyCustomModelSaverExporterCallback(tf.keras.callbacks.ModelCheckpoint):
  """
  Custom callback class for recording training history.

  """
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
           initial_value_threshold=None, #the last best monitored value that will allow for checkpoint triggering
           **kwargs):
    
    #init tf.keras.callbacks.ModelCheckpoint parent
    super(MyCustomModelSaverExporterCallback, self).__init__( filepath,
                                                              monitor,
                                                              verbose,
                                                              save_best_only,
                                                              save_weights_only,
                                                              mode,
                                                              save_freq,
                                                              initial_value_threshold=initial_value_threshold
                                                              )
    self.settings=settings
    self.settings.model_export_filename=os.path.join(os.getcwd(),'exported_models')
    self.previous_model_params=previous_model_params.copy() if previous_model_params is not None else None
    self.dst_metrics = METRIC_SET.get_metrics(["fb-trusted_dst", "fb-model_cosine_dst"])

  def get_config(self):
    """
    Returns the configuration of the callback.

    :return: Configuration dictionary.
    :rtype: dict
    """
    config=super(CustomHistory, self).get_config()
    config['settings']=self.settings
    config['previous_model_params']=self.previous_model_params
    
    return config

  def on_epoch_end(self, epoch, logs=None):
    """
    Called at the end of an epoch during training.

    :param epoch: Integer, index of the current epoch.
    :param logs: Dictionary, metric results for this epoch.
    """
    #call parent function
    print('on_epoch_end, saving checkpoint for round ', epoch)
    super(MyCustomModelSaverExporterCallback, self).on_epoch_end(epoch, logs)

    #log model change wrt previous epoch
    if self.previous_model_params is not None:
      self.model.track_weights_change(self.previous_model_params, epoch, prefix='epoch_local_model_change', dst_metrics=self.dst_metrics)
    self.previous_model_params=self.model.get_weights()
    if logs is None:
      print('WARNING, no logs dict is provided to ModelCheckpoint.on_epoch_end, checkpointing on best epoch will not work')
    
    if self.save_freq == 'epoch':
      try:
        if False:#self.model._in_multi_worker_mode():
          # FIXME/REMINDER Exclude training state variables in user-requested checkpoint file.
          with self._training_state.untrack_vars():
            self._export_model(epoch=epoch, logs=logs)
        else:
          self._export_model(epoch=epoch, logs=logs)
      except Exception as e:
        print('Model exporting failed for some reason',e)
    print('Epoch checkpoint save and export processes done.')

  def _export_model(self, epoch, logs=None):
    """
    Exports the model for serving and other purposes.

    :param epoch: Integer, index of the current epoch.
    :param logs: Dictionary, metric results for this epoch.
    """
    print('Exporting model...')
    current = logs.get(self.monitor)
    if current==self.best:
      print('Saving complete keras model thus enabling fitting restart as well as model load and predict')
      checkpoint_folder=os.path.join(os.getcwd(),'checkpoints', self.settings.cid, 'model_epoch_{version}'.format(version=epoch))
      self.model.save(checkpoint_folder)
      print('EXPORTING A NEW MODEL VERSION FOR SERVING')
      print('Exporting model at epoch {}.'.format(epoch))
      exported_module=self.settings.get_served_module(self.model, self.settings.model_name)
      if not(hasattr(exported_module, 'served_model')):
        raise ValueError('Experiment settings file MUST have \'served_model\' function with @tf.function decoration.')
      if self.settings.save_only_last_best_model:
        output_path=os.path.join(self.settings.model_export_filename, self.settings.cid,'1')
      else:
        output_path=os.path.join(self.settings.model_export_filename, self.settings.cid,str(epoch))
      
      model_concrete_function=exported_module.served_model.get_concrete_function()
      signatures={'serving_default':model_concrete_function, self.settings.model_name:model_concrete_function}
      try:
        module_graph= model_concrete_function
        print(module_graph.pretty_printed_signature(verbose=False))
          
        #module_graph[self.settings.model_name]=exported_module.served_model
        print('Exporting RAW serving model relying on tf.saved_model.save')
        tf.saved_model.save(
          exported_module,
          output_path,
          signatures=signatures,
          options=None
          )
        with open(os.path.join(self.settings.model_export_filename,'modelserving.signatures'), 'w') as f:
          f.write(module_graph.pretty_printed_signature())  
    
        if len(self.settings.used_gpu_IDs)==0:
          print('No GPU available to enable model export with TF-TensorRT')
        else:
          print('Now exporting model for inference with TensorRT...')
          try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            #print(os.listdir(output_path))
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
            if self.settings.enable_mixed_precision:
              #TODO, refine precision conversion, see : https://colab.research.google.com/github/vinhngx/tensorrt/blob/vinhn-tf20-notebook/tftrt/examples/image-classification/TFv2-TF-TRT-inference-from-Keras-saved-model.ipynb?hl=en#scrollTo=qKSJ-oizkVQY
              # and official doc https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#tf-trt-api-20
              conversion_params = conversion_params._replace(precision_mode="FP16")
            #print('Export for inference conversion_params:', conversion_params)
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
          #get more information here: https://www.tensorflow.org/lite/performance/post_training_quantization
          """
          FIXME: to be refined
          converter.optimizations = [tf.lite.Optimize.DEFAULT]
          converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                       tf.lite.OpsSet.TFLITE_BUILTINS]
          #define a tensorflow epresentative dataset from signatures:
          def representative_dataset():
            #making use of the identified signatures extracted previously
            for i in range(100):
              yield [tf.constant(0.0, shape=(1, self.settings.input_shape[0], self.settings.input_shape[1], self.settings.input_shape[2]))]
          converter.representative_dataset = representative_dataset
          """
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

#--------------------------------------------------------------
# Get the callbacks output into a history object 
#--------------------------------------------------------------

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
    """
    Callback function called at the beginning of training.
    Logs the start of a new training session.
    """
    print('******** HISTORY starting a training session...')
  
  def on_epoch_end(self, epoch, logs=None):
    """
    Callback function called at the end of each epoch.
    """
    #print('******** HISTORY on_epoch_end... logs=',logs)
    super(CustomHistory, self).on_epoch_end(epoch, logs)

  def get_config(self):
    """
    Returns the configuration of the callback.

    Returns:
        dict: Configuration dictionary.
    """
    return super(CustomHistory, self).get_config()

#--------------------------------------------------------------
# Prepare all standard callbacks as a dictionary
#--------------------------------------------------------------

class ImageSummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, object):
      if hasattr(object, 'get_image_summary'):
        self.object=object
      else:
        raise ValueError('passed object has no get_summary method')
    def on_epoch_end(self, epoch, logs=None):
        super(ImageSummaryCallback, self).on_epoch_end(epoch, logs)
        tf.summary.image('confusion_matrix', self.object.get_image_summary(), step=epoch)
        #tf.summary.image('gen_stats', self.object.gen_stats(), step=epoch) # to fix or to remove !
        #print(self.settings)

def define_callbacks(usersettings, 
                     model, 
                     val_data,
                     train_iterations_per_epoch, 
                     file_writer, 
                     log_dir, 
                     metrics=[],
                     previous_model_params=None, 
                     custom_callbacks:dict={}, 
                     initial_value_threshold=None,
                     monitored_loss_name_override=None
                     ):
  """
  Define callbacks for training a model.

  :param usersettings: The user settings object.
  :type usersettings: UserSettings

  :param model: The model to train.
  :type model: tf.keras.Model

  :param train_iterations_per_epoch: The number of training iterations per epoch.
  :type train_iterations_per_epoch: int

  :param file_writer: The file writer for TensorBoard logging.
  :type file_writer: tf.summary.FileWriter

  :param log_dir: The directory for log files.
  :type log_dir: str

  :param metrics: The set of metrics used along the optimisation process.
  :type initial_value_threshold: list, optional

  :param previous_model_params: Parameters from a previous model.
  :type previous_model_params: dict, optional

  :param custom_callbacks: Custom callbacks to be added.
  :type custom_callbacks: dict, optional

  :param initial_value_threshold: Threshold for initial value comparison.
  :type initial_value_threshold: float, optional

  :param monitored_loss_name_override: The name of the monitored loss to consider in place of usersettings.monitored_loss_name.
  :type monitored_loss_name_override: str, optional
  
  :return: The dictionary of callbacks.
  :rtype: dict
  """
  monitored_loss_name=usersettings.monitored_loss_name
  if monitored_loss_name_override:
    monitored_loss_name=monitored_loss_name_override

  all_callbacks={}
  #add the history callback
  all_callbacks['history_callback']=CustomHistory()#tf.keras.callbacks.History()
  # -> terminate on NaN loss values
  all_callbacks['TerminateOnNaN_callback']=tf.keras.callbacks.TerminateOnNaN()
  # -> apply early stopping
  early_stopping_patience=usersettings.early_stopping_patience if hasattr(usersettings, 'early_stopping_patience') else 5
  all_callbacks['earlystopping_callback']=tf.keras.callbacks.EarlyStopping(
                            monitor=monitored_loss_name,
                            patience=early_stopping_patience,
                            restore_best_weights=True
                          )
  all_callbacks['reduceLROnPlateau_callback']=tf.keras.callbacks.ReduceLROnPlateau(monitor=monitored_loss_name, factor=0.1,
                              patience=(early_stopping_patience*2)//3, min_lr=0.000001, verbose=True)

  #-> checkpoint each epoch
  all_callbacks['checkpoint_callback']=MyCustomModelSaverExporterCallback(#tf.keras.callbacks.ModelCheckpoint(
                                            os.path.join(os.getcwd(),'checkpoints/', usersettings.cid),
                                            usersettings,
                                            monitor=monitored_loss_name,
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            save_freq='epoch',
                                            previous_model_params=previous_model_params,
                                            initial_value_threshold=initial_value_threshold)

  if callable(usersettings.custom_tensorboard_logs):
    #print('Adding custom Tensorboard logger ON EPOCH END. keep some time, have a look at existing tools in tools/custom_display_tensorboard.py')
    with file_writer.as_default():
      def custom_logger_fn(epoch, logs):
        usersettings.custom_tensorboard_logs(model, epoch, logs)
      all_callbacks['on_epoch_end_custom_logger_callback']=tf.keras.callbacks.LambdaCallback(on_epoch_end=custom_logger_fn)
  #complete generic callbacks by user defined ones
  settings_addon_callback=usersettings.addon_callbacks(model, val_data, val_data)#FIXME: should addons callbacks receive the datasets ?
  if isinstance(settings_addon_callback, dict):
    all_callbacks.update(settings_addon_callback)
  else:
    raise ValueError('Addon callback specified in the experiment settings file should return a python dictionnary (at least an empty one)')

  #-> activate profiling if required
  profile_batch =0
  if usersettings.use_profiling is True and train_iterations_per_epoch>0:
    print('train_iterations_per_epoch', train_iterations_per_epoch)
    t_start=int(max(1,train_iterations_per_epoch//2-30))
    t_stop=int(t_start+min(60, train_iterations_per_epoch//4))
    if t_stop>train_iterations_per_epoch:
      t_stop=train_iterations_per_epoch-1
      print('too few iterations to perform a reliable profiling')
    profile_batch = '{t1},{t2}'.format(t1=t_start, t2=t_stop)
    print('Profiling is applied, for more details and log analysis, check : https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras')
  # -> set embeddings to be logged
  if not 'isRL' in usersettings.hparams.keys():
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

  if isinstance(custom_callbacks, dict):
     if len(custom_callbacks.keys())>0:
      print('Found custom callbacks, adding to default callbacks')
      all_callbacks.update(custom_callbacks)
  

  # look for metric callbacks
  # if any new custom metric has a target method that generates
  #  data to be logged on Tensorboard, the following loop look for them and adds the appropriate callbacks
  # For now, only  ImageSummaryCallback is called if a metric has the get_image_summary method
  def check_metric_maybe_add_callback(metrics_list):
    for metric in metrics_list:
      if 'get_image_summary' in dir(metric):#hasattr(metric, 'get_image_summary'):
        print('Found metric with required ImageSummaryCallback callback:',metric)
        all_callbacks[metric.name]=ImageSummaryCallback(metric)
  
  # check and adapt: metrics should be either :
  # -> a single metric object or a list of metrics in case of a single output model
  # -> or a dictionary of single metric object or lists of metrics in case of a multi output model
  if isinstance(metrics, list):
    check_metric_maybe_add_callback(metrics)
  elif isinstance(metrics, dict):
    for model_output in metrics.keys():
      if isinstance(metrics[model_output], list):
        check_metric_maybe_add_callback(metrics[model_output])
      elif issubclass(metrics[model_output].__class__, tf.keras.metrics.Metric):
        check_metric_maybe_add_callback([metrics[model_output]])
  
  elif issubclass(metrics.__class__, tf.keras.metrics.Metric):
    check_metric_maybe_add_callback([metrics])
  
  return all_callbacks
