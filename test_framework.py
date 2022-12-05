''' a basic script that tries to start each of the demo scripts
TODO:move to unit testing with pytest
'''

import pytest
experiment_setting_files_to_test=[
                # basic scripts that do not require additionnal data (either already with the sources or downloaded automatically
                #baby test, NO hyperparameters
                {'script':'examples/regression/mysettings_curve_fitting.py', 'hparams':{}},
                {'script':'examples/regression/mysettings_curve_fitting_concrete_dropout.py', 'hparams':{}},
                #baby test, WITH hyperparameters
                {'script':'examples/regression/mysettings_curve_fitting.py', 'hparams':{'nbEpoch':2}},
                #scripts sensitive to tensorflow updates...
                {'script':'examples/embedding/mysettings_1D_experiments.py', 'hparams':{'nbEpoch':2}},
                {'script':'examples/generative/mysettings_began.py', 'hparams':{'nbEpoch':2}},
                

                #WARNING : the following depend on some specific data that have to be downloaded and targetted in the settings script:
                #{'script':'examples/segmentation/mysettings_semanticSegmentation.py', 'hparams':{'nbEpoch':2}},
                #FIXME:the premade estimator based model in examples/regression/mysettings_curve_fitting_premade_estimator.py impact on the following tests, possible cause : usersettings global variable in experiments_manager.py should be removed
                #{'script':'examples/regression/mysettings_curve_fitting_premade_estimator.py', 'hparams':None},

                {'script':'private/mysettings_cycle_constraints_trades_bayesian.py'},

                {'script':'private/mysettings_heterogeneous_data_fusion.py', 'hparams':{}},
                #{'script':'private/mysettings_embeddings_hyperspectral_images.py', 'hparams':{'nbEpoch':2}},
                {'script':'private/mysettings_embeddings_mnist.py', 'hparams':{'nbEpoch':2}},
                {'script':'private/mysettings_embeddings_multispectral_images.py', 'hparams':{'nbEpoch':2}},
                #FIXME : this test works if run in the end but makes following tests fail. This shows that the global variable 'usersettings' in experiments_manager.py should be removed. This will be possible by refactoring code into Object Programming imposed by function model_fn that has a fixed number of parameters
                {'script':'examples/regression/mysettings_curve_fitting_premade_estimator.py', 'hparams':{}}

                ]


# a basic object that reports some flags required by the experiments_manager script
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

