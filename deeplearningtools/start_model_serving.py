# ========================================
# FileName: start_model_serving.py
# Date: 29 june 2020 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of tools to help running tensorflow model server
# for DeepLearningTools.
# =========================================
r"""
Use example : start model server on an experiment using command:
  -> note the use of the optionnal -psi option that relies on tensorflow_model_server installed in a singularity container (build from definition file tf_server.def provided in the repository)
  python3 start_model_serving.py --model_dir=/home/alben/workspace/listic-deeptool/experiments/examples/curve_fitting/my_test_hiddenNeurons50_predictSmoothParamsTrue_learningRate0.1_nbEpoch5000_addNoiseTrue_anomalyAtX-3_2020-03-10--21\:59\:52/ -psi /home/alben/install/containers/tf_server.cpu.sif
"""

import os
import glob
import argparse
import deeplearningtools.helpers.model_serving_tools as srv_comm_tools

def get_served_model_info(target_model_path, expected_model_name, tf_server_container_path):
  """
  Basic function that checks served model behaviors.

  :param target_model_path: The path to a servable model directory
  :type target_model_path: str
  :param expected_model_name: The model name that is expected to be found on the server
  :type expected_model_name: str
  :param tf_server_container_path: the path to an apptainer (ex. singularity) container.
  :type tf_server_container_path: str
  
  :return: None
  """
  import subprocess
  #get the first subfolder of the served models directory
  served_model_info_cmd='saved_model_cli show --dir {target_model} --tag_set serve --signature_def {model_name}'.format(target_model=target_model_path,
                                                                                      model_name=expected_model_name)
  if len(tf_server_container_path)>0:
    served_model_info_cmd='apptainer exec '+tf_server_container_path+' '+served_model_info_cmd
  print('Checking served model available signatures using command '+served_model_info_cmd)
  cmd_result=subprocess.check_output(served_model_info_cmd.split())
  print('Answer:')
  print(cmd_result.decode())
  if expected_model_name in cmd_result.decode():
    print('Target model {target} name found in the command answer'.format(target=expected_model_name))
  else:
    raise ValueError('Target model {target} name NOT found in the command answer'.format(target=expected_model_name))

def start_model_serving(flags):
  """
  Starts a TensorFlow model server for serving trained models, get trained model config, check server, and start it from host of container.
  """
  # get trained model config
  config=srv_comm_tools.get_model_server_cfg(flags.model_dir)
  print('model_name', config['SERVER']['model_name'])
  #look for a model in the directory
  model_folder=os.path.join(flags.model_dir,'exported_models')
  print('Looking for trained models in ',model_folder)
  found_models=glob.glob(os.path.join(model_folder,'*/'))
  if len(found_models)==0:
    raise ValueError('No servable model found in '+found_models)
  
  print('Found', len(found_models),' servable models in ',model_folder, ':', found_models)
  if FLAGS.version >0:
    if FLAGS.version<len(found_models):
      print('Multimodel case study (certainly federated learning clients), attempting to load a specific model version :', FLAGS.version)
      target_model_path=os.path.join(model_folder, str(FLAGS.version))
    else:
      raise ValueError('Requested version '+str(FLAGS.version)+' not found in '+model_folder)
  else:
    target_model_path=model_folder

  #target_model_path=os.path.join(model_folder, target_model)
  if not(os.path.exists(target_model_path)):
    raise ValueError('served models directory not found : '+target_model_path)
  print('Found at least one servable model directory '+str(target_model_path))
  
  #check server
  try:
    get_served_model_info(target_model_path, config['SERVER']['model_name'], flags.tf_server_container_path)
  except Exception as e:
    print('Could not call saved_model_cli, is Tensorflow installed? Error:', e)
  tensorflow_start_cmd=" --port={port} --model_name={model} --model_base_path={model_dir}".format(port=config['SERVER']['port'],
                                                                                      model=config['SERVER']['model_name'],
                                                                                      model_dir=target_model_path)
  #start server from host of singularity container
  if len(flags.tf_server_container_path)>0:
    print('Starting Tensorflow model server from provided singularity container : '+FLAGS.tf_server_container_path)
    tensorflow_start_cmd='apptainer run --nv '+FLAGS.tf_server_container_path+tensorflow_start_cmd
  else:
    print('Starting Tensorflow model server installed on system')
    tensorflow_start_cmd='tensorflow_model_server '+tensorflow_start_cmd
  
  print('Starting tensorflow server with command :'+tensorflow_start_cmd)
  os.system(tensorflow_start_cmd)

if __name__ == "__main__":

    #get command line parameters or use defaults
    parser = argparse.ArgumentParser(description='demo_semantic_segmentation')
    parser.add_argument("-m","--model_dir", default=None,
                        help="Output directory for model and training stats.")
    parser.add_argument("-psi","--tf_server_container_path", default='',
                        help="start the tensorflow server on a singularity container to run predictions")
    parser.add_argument("-v","--version", default=0, type=int,
                        help="specify a specific version of the model to serve")
    parser.add_argument("-c","--commands", action='store_true',
                        help="show command examples")

    FLAGS = parser.parse_args()

    start_model_serving(FLAGS)