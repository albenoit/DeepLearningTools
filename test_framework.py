# ========================================
# FileName: test_framework.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A basic script that tries to start each of the demo scripts
# for DeepLearningTools.
#TODO:move to unit testing with pytest
# =========================================

import pytest
import tensorflow as tf
from deeplearningtools.tools.command_line_parser import get_default_args
from deeplearningtools.tools.experiment_settings import  loadExperimentsSettings, loadModel_def_file, SETTINGSFILE_COPY_NAME
from deeplearningtools import experiments_manager
from deeplearningtools import start_kafka_producer
from deeplearningtools import start_federated_server
from deeplearningtools.helpers import tensor_msg_io  
from deeplearningtools.tools import experiments_settings_surgery
import os

#basic function that just starts a demo script
def start_training_script(FLAGS, experiment_settings_file, experiment_settings_hparams):
  global jobState
  global jobSessionFolder
  global loss
  print('************ CWD=', os.getcwd())
  
  job_result = experiments_manager.run(FLAGS, train_config_script=experiment_settings_file, external_hparams=experiment_settings_hparams)
  print('trial ended successfully, output='+str(job_result))
  if 'loss' in job_result.keys():
    loss=job_result['loss']
  jobSessionFolder=job_result['sessionFolder']
  jobState=True
  return jobState, jobSessionFolder, loss

def test_kafka_producer_no_broker():
  import kafka
  with pytest.raises(kafka.errors.NoBrokersAvailable) as ce:
    FLAGS = start_kafka_producer.get_commands().parse_args([])
    FLAGS.usersettings='examples/regression/mysettings_curve_fitting.py'
    print(FLAGS)
    start_kafka_producer.run(FLAGS)

def test_tensor_msg_io():
  input_constant=tf.constant([0.1, 2.03], dtype=tf.float16)
  input_label=b'goat'
  #encode
  serialized_example_2 = tensor_msg_io.serialize_tensor_with_label(input_constant, input_label)
  #decode
  example_proto=tf.constant(serialized_example_2)
  decoded_example=tensor_msg_io.decode_tensor_with_label_example(example_proto, tf.float16)
  print('unserialized example_2', decoded_example)
  assert   (decoded_example[0]==input_label).numpy()==True and (decoded_example[1]==input_constant).numpy()[0]==True

def test_basic_regression_noHparams():
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/regression/mysettings_curve_fitting.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_training_script(FLAGS, FLAGS.usersettings, {})
  print('Test end, loss=', loss)
  assert loss < 10

def test_basic_regression_epochHparams():
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/regression/mysettings_curve_fitting.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_training_script(FLAGS, FLAGS.usersettings, {'nbEpoch':2})
  print('Test end, loss=', loss)
  assert loss < 50

def test_timeseries():
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/timeseries/mysettings_timeseries_forecasting.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_training_script(FLAGS, FLAGS.usersettings, {'nbEpoch':2})
  print('Test end, loss=', loss)
  assert loss < 500

def test_timeseries_temporian():
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/timeseries/mysettings_timeseries_forecasting_temporian.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_training_script(FLAGS, FLAGS.usersettings, {'nbEpoch':2})
  print('Test end, loss=', loss)
  assert loss < 2500

def test_classification():
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/classification/mysettings_image_classification.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_training_script(FLAGS, FLAGS.usersettings, {'nbEpoch':2})
  print('Test end, loss=', loss)
  assert loss < 2

def tRest_semantic_segmentation():
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/segmentation/mysettings_semanticSegmentation.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_training_script(FLAGS, FLAGS.usersettings, {'nbEpoch':2})
  print('Test end, loss=', loss)
  assert loss < 2

def test_basic_1Dembedding():
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/embedding/mysettings_1D_experiments.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_training_script(FLAGS, FLAGS.usersettings, {'nbEpoch':2})
  print('Test end, loss=', loss)
  assert loss < 500

def test_flower_simulation():
  expected_nb_rounds=1
  FLAGS = start_federated_server.get_commands().parse_args([])
  FLAGS.usersettings='examples/federated/mysettings_curve_fitting.py'
  FLAGS.simulation=True
  FLAGS.num_rounds=expected_nb_rounds
  # run federated experiment in simulation mode
  result=start_federated_server.run(FLAGS)#, {'session_number':0, 'session_folder':sessionFolder})
  assert len(result.metrics_centralized['val_loss']) == expected_nb_rounds+1


def test_flower_simulation_listiccfl():
  expected_nb_rounds=1
  FLAGS = start_federated_server.get_commands().parse_args([])
  FLAGS.usersettings='examples/federated/mysettings_curve_fitting.py'
  FLAGS.agregation='ListicCFL_strategy'
  FLAGS.simulation=True
  FLAGS.num_rounds=expected_nb_rounds
  # run federated experiment in simulation mode
  result=start_federated_server.run(FLAGS)#, {'session_number':0, 'session_folder':sessionFolder})
  assert len(result.metrics_centralized['val_loss']) == expected_nb_rounds+1



if __name__ == "__main__":
  print('Starting some test functions as demos, please consider running pytest like this :')
  print(' pytest test_framework.py OR FROM A CONTAINER, apptainer exec path/to/tf2_addons.sif pytest test_framework.py')
  # manually choose a given test
  test_flower_simulation()
  test_basic_1Dembedding()
  test_basic_regression_noHparams()
  test_basic_regression_epochHparams()
  test_timeseries()
  test_timeseries_temporian()
  test_classification()
  print("END")
