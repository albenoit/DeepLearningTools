''' A set of tools to help running tensorflow model server
  Alexandre Benoit, LISTIC, 2020

  Use example : start model server on an experiment using command:
  -> note the use of the optionnal -psi option that relies on tensorflow_model_server installed in a singularity container (build from definition file tf_server.def provided in the repository)
  python3 start_model_serving.py --model_dir=/home/alben/workspace/listic-deeptool/experiments/examples/curve_fitting/my_test_hiddenNeurons50_predictSmoothParamsTrue_learningRate0.1_nbEpoch5000_addNoiseTrue_anomalyAtX-3_2020-03-10--21\:59\:52/ -psi /home/alben/install/containers/tf_server.cpu.sif

'''
import os
import argparse
import configparser

def get_served_model_info(one_model_path, expected_model_name, singularity_tf_server_container_path):
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
  if len(singularity_tf_server_container_path)>0:
    served_model_info_cmd='singularity exec '+singularity_tf_server_container_path+' '+served_model_info_cmd
  print('Checking served model available signatures using command '+served_model_info_cmd)
  cmd_result=subprocess.check_output(served_model_info_cmd.split())
  print('Answer:')
  print(cmd_result.decode())
  if expected_model_name in cmd_result.decode():
    print('Target model {target} name found in the command answer'.format(target=expected_model_name))
  else:
    raise ValueError('Target model {target} name NOT found in the command answer'.format(target=expected_model_name))


def start_model_serving(flags):
  model_dir=flags.model_dir
  config_filename='model_serving_setup.ini'
  print('### START TENSORFLOW SERVER on experiment : ', model_dir, '###')

  model_folder=os.path.join(model_dir,'exported_models')
  print('Considering served model parent directory:'+model_folder)
  config_file=os.path.join(model_dir, config_filename)
  print('Reading config file : ', config_file)
  #load server config
  if os.path.exists(config_file) == False:
    raise ValueError("COnfig file does not exist")
  config=configparser.ConfigParser()
  config.read(config_file, encoding='utf8')
  print(config.keys())
  print('CONFIG:', config)
  print('model_name', config['SERVER']['model_name'])
  #look for a model in the directory
  one_model=next(os.walk(model_folder))[1][0]
  one_model_path=os.path.join(model_folder, one_model)
  if not(os.path.exists(one_model_path)):
    raise ValueError('served models directory not found : '+one_model_path)
  print('Found at least one servable model directory '+str(one_model_path))
  
  #check server
  
  try:
    get_served_model_info(one_model_path, config['SERVER']['model_name'], flags.singularity_tf_server_container_path)
  except Exception as e:
    print('Could not call saved_model_cli, is Tensorflow installed? Error:', e)
  tensorflow_start_cmd=" --port={port} --model_name={model} --model_base_path={model_dir}".format(port=config['SERVER']['port'],
                                                                                      model=config['SERVER']['model_name'],
                                                                                      model_dir=model_folder)
  #start server from host of singularity container
  if len(flags.singularity_tf_server_container_path)>0:
    print('Starting Tensorflow model server from provided singularity container : '+FLAGS.singularity_tf_server_container_path)
    tensorflow_start_cmd='singularity run --nv '+FLAGS.singularity_tf_server_container_path+tensorflow_start_cmd
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
    parser.add_argument("-psi","--singularity_tf_server_container_path", default='',
                        help="start the tensorflow server on a singularity container to run predictions")
    parser.add_argument("-c","--commands", action='store_true',
                        help="show command examples")

    FLAGS = parser.parse_args()

    start_model_serving(FLAGS)
