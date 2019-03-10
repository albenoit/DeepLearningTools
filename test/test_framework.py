''' a basic script that tries to start each of the demo scripts
TODO:move to unit testing with pytest
'''
experiment_setting_files_to_test=[
                # basic scripts that do not require additionnal data (either already with the sources or downloaded automatically
                #baby test, NO hyperparameters
                {'script':'examples/regression/mysettings_curve_fitting.py', 'hparams':None},
                {'script':'examples/regression/mysettings_curve_fitting_concrete_dropout.py', 'hparams':None},
                #baby test, WITH hyperparameters
                {'script':'examples/regression/mysettings_curve_fitting.py', 'hparams':{'nbEpoch':2}},
                #scripts sensitive to tensorflow updates...
                {'script':'examples/embedding/mysettings_1D_experiments.py', 'hparams':{'nbEpoch':2}},
                {'script':'examples/generative/mysettings_began.py', 'hparams':{'nbEpoch':2}},


                #WARNING : the following depend on some specific data that have to be downloaded and targetted in the settings script:
                {'script':'examples/segmentation/mysettings_semanticSegmentation.py', 'hparams':{'nbEpoch':2}},
                #FIXME:the premade estimator based model in examples/regression/mysettings_curve_fitting_premade_estimator.py impact on the following tests, possible cause : usersettings global variable in experiments_manager.py should be removed
                {'script':'examples/regression/mysettings_curve_fitting_premade_estimator.py', 'hparams':None},

                ]

#basic function that just starts a demo script
def start_script(experiment_settings_file, experiment_settings_hparams):
  global jobState
  global jobSessionFolder
  global loss
  import experiments_manager
  job_result = experiments_manager.run(experiment_settings_file, experiment_settings_hparams)
  print('trial ended successfully, output='+str(job_result))
  if 'loss' in job_result.keys():
    loss=job_result['loss']
  jobSessionFolder=job_result['sessionFolder']
  jobState=True
  return jobState, jobSessionFolder, loss

def script_tester(test_dict, failOnError=False):#False):
  '''starts a running session for a given experiment setup
  Arg:
    test_dict: the experiment script file to run with
    failOnError: set True if the function should crash on any error instead of catching the error and keep running
  '''
  experiment_settings_file=test_dict['script']
  experiment_settings_hparams=None
  if 'hparams' in test_dict:
    experiment_settings_hparams=test_dict['hparams']
  print('*** Testing script : '+experiment_settings_file)
  #start the training session
  global jobState
  global jobSessionFolder
  global loss
  #init states
  jobState=False
  jobSessionFolder=None
  loss=1e10
  # start script
  try:
    '''import threading
    thread = threading.Thread(target=start_script, args=[experiment_settings_file, experiment_settings_hparams])
    thread.start()
    thread.join()
    '''
    jobState, jobSessionFolder, loss=start_script(experiment_settings_file, experiment_settings_hparams)
  except Exception as e:
    print('Job failed for some reason:'+str(e))
    if failOnError:
      raise ValueError('Failed to excetute test {test}, error={error}'.format(test=test_dict, error=e))
  report=test_dict
  report.update({'success': jobState, 'jobSessionFolder':jobSessionFolder, 'loss': loss})

  return report

import pandas as pd
test_report_list=[]

for id, test_case in enumerate(experiment_setting_files_to_test):
  print('***************************************************************')
  print('Test case:'+str(test_case))
  script_output=script_tester(test_case)
  script_output.update({'id':id})
  test_report_list.append(script_output)

print('***************************************************************')
print('***************************************************************')
print('tests finished, report:')
success_count=0
for id, report in enumerate(test_report_list):
  print('*** Test {tid} reported: {rep}'.format(tid=id, rep=report))
  if report['success']:
    success_count+=1
  else:
    print('     --> test failed, try running manually script as follows: python experiments_manager.py --usersettings='+report['script'])
print('### Number of successful test(s) {jobOK} vs total {alltrials}'.format(jobOK=success_count,
                                                                      alltrials=len(test_report_list)))
