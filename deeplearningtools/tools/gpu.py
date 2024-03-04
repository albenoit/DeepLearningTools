### a set of tools related to GPU management
import os
import tensorflow as tf
# Custom generic functions applied when running experiments
def check_GPU_available(usersettings):
  """check GPU requirements vs availability: if usersettings.used_gpu_IDs is not empty, then check GPU availability accordingly
     Args:
      usersettings, the experiments settings defined by the user
     Raises SystemError if no GPU available
  """
  gpu_workers_nb=0
  print('*** GPU devices detection ***')
  # let ensorFlow automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
  tf.config.set_soft_device_placement(True)
  if len(usersettings.used_gpu_IDs)>0:
    device_name = tf.test.gpu_device_name()
    print('Found GPU at: {}'.format(device_name))
    
    gpus = tf.config.list_physical_devices('GPU')

    #-> first check availability
    if len(gpus) ==0 and len(usersettings.used_gpu_IDs):
      print('Could not find any GPU, trying to reload driver...')
      #-> first try to wake it up
      os.system("nvidia-modprobe -u -c=0")
      gpus = tf.config.list_physical_devices('GPU')
      if len(gpus) ==0 and len(usersettings.used_gpu_IDs):
        print('No GPU found')
        raise SystemError('Required GPU(s) not found')

    print('Found GPU devices:', gpus)
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      visible_devices=[gpus[id] for id in usersettings.used_gpu_IDs]
      print('Setting visible devices:',visible_devices)
      try:
        tf.config.set_visible_devices(visible_devices, 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')

        for gpuID in range(len(gpus)):
          print('Found GPU:', gpuID)
          #tf.config.experimental.set_memory_growth(gpus[gpuID], True)
          if tf.test.gpu_device_name() != '/device:GPU:0':
            print('WARNING: GPU device not found.')
          else:
            print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))
          tf.config.experimental.set_memory_growth(gpus[gpuID], True)
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        gpu_workers_nb=len(logical_gpus)

      except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
  else:
    print('No GPU required for this experiment (usersettings.used_gpu_IDs is empty)')
  return gpu_workers_nb

def get_available_gpus():
    """return the list of available GPUs
    """
    gpus = tf.config.list_physical_devices('GPU')
    return gpus

def check_mixed_precision_compatibility(model, usersettings):
    #check if mixed precision compatibility are satisfied: https://docs.nvidia.com/deeplearning/performance/index.html
    with open(usersettings.sessionFolder+'/model_mixed_precision_compatibility.info', 'w')as f:
        if usersettings.batch_size%64!=0:
            f.write('Suboptimal batch size (should be a multiple of 64 to get efficient tiling and reduced overhead') 
        for i,layer in enumerate(model.layers[1:]):
            f.write('\n=== layer'+str(i)+str(layer))
            try:
                f.write('\n   layer.input_shape[-1], layer.input_shape[-1]'+str((layer.input_shape[-1], layer.output_shape[-1])))
                if ((layer.input_shape[-1])%8 != 0) or ((layer.output_shape[-1])%8 != 0):
                    f.write('\n    -> /!\ Layer not tensorcore compliant (index, name, input, output):'+str((i,layer.name,layer.input_shape, layer.output_shape)))
            except Exception as e:
                f.write('\n    -> /!\ Could not check layer:'+str((i, layer))+str(e))
    
