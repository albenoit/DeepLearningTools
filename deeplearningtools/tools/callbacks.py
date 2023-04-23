import os
from tensorboard.plugins.hparams import api as hp

import tensorflow as tf

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
    self.previous_model_params=previous_model_params

  def get_config(self):
    config=super(CustomHistory, self).get_config()
    config['settings']=self.settings
    config['previous_model_params']=self.previous_model_params
    
    return config

  def on_epoch_end(self, epoch, logs=None):
    #call parent function
    print('on_epoch_end, saving checkpoint for round ', epoch)
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
          # FIXME/REMINDER Exclude training state variables in user-requested checkpoint file.
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
  def get_config(self):
     return super(CustomHistory, self).get_config()

#####################################
# prepare all standard callbacks as a dictionary
def define_callbacks(usersettings, model, train_iterations_per_epoch, file_writer, log_dir, previous_model_params=None, custom_callbacks:dict={}, initial_value_threshold=None):
  all_callbacks={}
  #return all_callbacks
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
                                            os.path.join(os.getcwd(),'checkpoints/', usersettings.cid),
                                            usersettings,
                                            monitor=usersettings.monitored_loss_name,
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            save_freq='epoch',
                                            previous_model_params=previous_model_params,
                                            initial_value_threshold=initial_value_threshold)


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
  if usersettings.use_profiling==True and train_iterations_per_epoch>0:
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

  if isinstance(custom_callbacks, dict):
     if len(custom_callbacks.keys())>0:
      print('Found custom callbacks, adding to default callbacks')
      all_callbacks.update(custom_callbacks)

  return all_callbacks
