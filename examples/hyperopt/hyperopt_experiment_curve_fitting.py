''' A basic example to launch a set of neural network training sessions to seek for the
best hyperparameters set that minimize a target criteria

@brief  based on hyperopt, start several training sessions concudted by the
script experiments_manager.py.
All trials are versionned as for any manually started single training session
with experiments_manager.py script.
The tuned hyperparameter values are stored in each trial folder as a python script :
-> 'experiment_settings_additionnal_hyperparameters.py'
-> this script is automatically loaded for training, validation, session restart on failure and serving/prediction

@author Alexandre Benoit, LISTIC, France

@notes: doc here https://github.com/hyperopt/hyperopt/wiki/FMin
-> a tutorial here : https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce
-> some ideas to study the hyperparameter and loss behaviors : https://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html

-> advanced method here : https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
'''

#basic parameters of the Hyperopt experiment
MAX_TRIALS = 100 # the maximum number of optimisation attempts
experiment_settings_file='examples/regression/mysettings_curve_fitting.py'
outputlog_folder = 'examples/hyperopt/hyperopt_experiments_curve_fitting'
toobig_loss=1e6 #the default trial loss value in case of job failure

# define a search space
from hyperopt import hp
space = {
        'hiddenNeurons': hp.randint('hiddenNeurons', upper=20)+1,#test between 1 and 21 neurons
        'learningRate': hp.uniform('learningRate', 1e-2, 1e-5),
        }


##### Hyperopt code starts here
#librairies imports

import matplotlib as mpl
mpl.use('Agg') #force not to display plot but rather write them to file
from hyperopt import fmin, tpe, space_eval, Trials, STATUS_FAIL, STATUS_OK, plotting
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import os

#ensure log dir exists
if not(os.path.exists(outputlog_folder)):
  os.makedirs(outputlog_folder)

# define the function that launches a single experiment trial
def single_experiment(hparams):
  print('*** Hyperopt : New experiment trial with hparams='+str(hparams))
  loss=toobig_loss
  jobState=STATUS_FAIL
  jobSessionFolder=None
  training_iterations=0
  #start the training session
  try:
    import experiments_manager
    job_result = experiments_manager.run(experiment_settings_file, hparams)
    print('trial ended successfully, output='+str(job_result))
    loss=job_result['loss']
    jobSessionFolder=job_result['sessionFolder']
    jobState=STATUS_OK
    training_iterations=job_result['global_step']
  except Exception as e:
    print('Job failed for some reason:'+str(e))

  return  {'loss': loss, 'status': jobState, 'jobSessionFolder':jobSessionFolder, 'training_iterations':training_iterations}

# minimize the objective over the space
trials = Trials()
try:
  best = fmin(single_experiment, space, algo=tpe.suggest, trials=trials, max_evals=MAX_TRIALS)
  #print('best='+str(best)) #WARNING, displayed hyperparameters do not take into account custom changes (+1, etc. in the space description)
  print('space_eval='+str(space_eval(space, best)))

except Exception as e:
  print('Hyperopt experiment interrupted not finish its job with error message:'+str(e))

best_trials = sorted(trials.results, key=lambda x: x['loss'], reverse=False)
print('best_trials:')
print(best_trials)

#experiments trials visualisation
#=> plot the evolution of each of the hyperparameters along the trials
import matplotlib.pyplot as plt
import pandas as pd
chosen_hparams_history=[]
finished_jobs=0
for trial in trials.trials:
  if trial['result']['status']==STATUS_OK:
    finished_jobs+=1
  current_trial_desc={'tid':trial['tid'],
                      'loss':trial['result']['loss'],
                      'training_iterations':trial['result']['training_iterations']}
  current_trial_desc.update(trial['misc']['vals'])
  trial_summary=pd.DataFrame(current_trial_desc)
  chosen_hparams_history.append(trial_summary)
trials_summary = pd.concat(chosen_hparams_history)

#write logs
print('*** Number of finished trials {jobOK} / total {alltrials}'.format(jobOK=finished_jobs,
                                                                      alltrials=len(trials.trials)))
trials_log_file=os.path.join(outputlog_folder,'hyperopt_curve_fitting_trials_report.csv')
trials_summary.to_csv(trials_log_file)
print('-> trials summary report written here : '+trials_log_file)

if finished_jobs>0:
  trials_summary.plot(subplots=True, x='tid')
  plt.legend(loc='best')
  plt.savefig(os.path.join(outputlog_folder, 'hparams_vs_trials'))
  #draw the scatter matrix
  scatter_matrix(trials_summary[list(trial['misc']['vals'])+['loss']], alpha=0.2, figsize=(6, 6), diagonal='kde')
  plt.savefig(os.path.join(outputlog_folder, 'scatter_matrix'))
  plt.figure()
  plotting.main_plot_history(trials)
  plt.savefig(os.path.join(outputlog_folder, 'hyperopt_history'))
  plt.figure()
  plotting.main_plot_histogram(trials)
  plt.savefig(os.path.join(outputlog_folder, 'hyperopt_histogram'))
  #plotting.main_plot_vars(trials, scope)#FIXME, check https://github.com/hyperopt/hyperopt/blob/master/hyperopt/tests/test_plotting.py
  try:
    plt.show()
  except:
    print('matplotlib.pyplot.show() could not be run, maybe this computer does not have a X server')
    pass
else:
  print('-> Could not run any trial correctly... look for errors, first try to start a single experiment manually')

try:
  #python3 commands...
  from hyperopt import graphviz
  graphviz_space_description=graphviz.dot_hyperparameters(space)
  open(os.path.join(outputlog_folder, 'hyperopt_graph.dot'), 'w').write(graphviz_space_description)
  #later in command line, type
  #graphviz_to_png_command='dot -Tpng hyperopt_graph.dot > hyperopt_graph.png && eog hyperopt_graph.png'
except Exception as e:
  print('Could not run some python 3 compatible commands, error='+str(e))
  pass
