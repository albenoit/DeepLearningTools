''' a basic script that tries to start each of the demo scripts
TODO:move to unit testing with pytest
'''

import pytest
from tools.command_line_parser import get_default_args

#basic function that just starts a demo script
def start_script(FLAGS, experiment_settings_file, experiment_settings_hparams):
  global jobState
  global jobSessionFolder
  global loss

  import experiments_manager
  job_result = experiments_manager.run(FLAGS, train_config_script=experiment_settings_file, external_hparams=experiment_settings_hparams)
  print('trial ended successfully, output='+str(job_result))
  if 'loss' in job_result.keys():
    loss=job_result['loss']
  jobSessionFolder=job_result['sessionFolder']
  jobState=True
  return jobState, jobSessionFolder, loss

def test_basic_regression_noHparams():
  #with pytest.raises(AttributeError) as ce:
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/regression/mysettings_curve_fitting.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_script(FLAGS, FLAGS.usersettings, {})
  print('Test end, loss=', loss)
  assert loss < 10

def test_basic_regression_epochHparams():
  #with pytest.raises(AttributeError) as ce:
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/regression/mysettings_curve_fitting.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_script(FLAGS, FLAGS.usersettings, {'nbEpoch':2})
  print('Test end, loss=', loss)
  assert loss < 50

def test_timeseries():
  #with pytest.raises(AttributeError) as ce:
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/timeseries/mysettings_timeseries_forecasting.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_script(FLAGS, FLAGS.usersettings, {'nbEpoch':2})
  print('Test end, loss=', loss)
  assert loss < 500

def test_classification():
  #with pytest.raises(AttributeError) as ce:
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/classification/mysettings_image_classification.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_script(FLAGS, FLAGS.usersettings, {'nbEpoch':2})
  print('Test end, loss=', loss)
  assert loss < 2

def test_basic_1Dembedding():
  #with pytest.raises(AttributeError) as ce:
  FLAGS = get_default_args()
  FLAGS.usersettings='examples/embedding/mysettings_1D_experiments.py'
  print(FLAGS)
  jobState, jobSessionFolder, loss = start_script(FLAGS, FLAGS.usersettings, {'nbEpoch':2})
  print('Test end, loss=', loss)
  assert loss < 500

if __name__ == "__main__":
  print('Starting some test functions as demos, please consider running pytest')
  # manually choose a given test
  test_basic_regression_noHparams()
  test_basic_regression_epochHparams()
  test_timeseries()
  test_classification()
  test_basic_1Dembedding()

