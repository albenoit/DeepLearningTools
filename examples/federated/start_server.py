# main parameter server, should be started first
import flwr as fl
import experiments_manager
import tensorflow as tf
import os

# retreive command line arguents
import argparse
parser = argparse.ArgumentParser(description='Deep learning experiments manager')
parser.add_argument("-u","--usersettings",
                    help="filename of the settings file that defines an experiment")
parser.add_argument("-w","--pretrainedmodelcheckpointpath", default="",
                    help="path to a previous model checkpoint that provides pretrained model weights")

FLAGS = parser.parse_args()

#get experiment settings filename path
usersettings_file=FLAGS.usersettings
#load experiments settings
def loadModel_def_file(usersettings):
  ''' basic method to load the model targeted by usersettings.model_file
  Args: sessionFolder, the path to the model file
  Returns: a keras model
  '''
  import imp
  model_path=usersettings.model_file
  try:
    model_def=imp.load_source('model_def', model_path)#importlib.import_module("".join(model_path.split('.')[:-1]))#
  except Exception as e:
    raise ValueError('loadModel_def_file: Failed to load model file {model} from sessionFolder {sess}, error message={err}'.format(model=usersettings.model_file, sess=sessionFolder, err=e))
  model=model_def.model

  print('loaded model file {file}'.format(file=model_path))
  return model


usersettings, sessionFolder = experiments_manager.loadExperimentsSettings(usersettings_file, isServingModel=False)
min_fl_clients_nb=0
if 'flClients' in usersettings.hparams:
  min_fl_clients_nb=usersettings.hparams['flClients']
else:
  raise ValueError('Experiment settings does not specify the minimum number of clients (hparams[\'flClients\']')
strategy_name=None
if 'federated' in usersettings.hparams:
  strategy_name=usersettings.hparams['federated']
else:
  raise ValueError('Experiment settings does not specify the minimum number of clients (hparams[\'flClients\']')
  
if os.path.exists(FLAGS.pretrainedmodelcheckpointpath):
  print('centralized model is pretrained... loading from ', FLAGS.pretrainedmodelcheckpointpath)
  from helpers.model import load_model
  model=load_model(FLAGS.pretrainedmodelcheckpointpath+'/checkpoints/')
else:
  print('Loading a clean model to be trained from scratch')
  model=loadModel_def_file(usersettings)(usersettings)

#load model as will be done by the clients taking into account the experiment settings


"""
REMINDER from https://flower.dev/docs/strategies.html: 
import flwr as fl

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,  # Sample 10% of available clients for the next round
    min_fit_clients=10,  # Minimum number of clients to be sampled for the next round
    min_available_clients=80,  # Minimum number of clients that need to be connected to the server before a training round can start
)
fl.server.start_server(config={"num_rounds": 3}, strategy=strategy)
"""

#first create or load a default model (the initialization of the weights that will be distributed to each client)
#get the init central model parameters to be distributed to each clients
init_params=fl.common.weights_to_parameters(model.get_weights())

#then apply the specified agregation strategy
print('Experiment settings file reports federated learning strategy:', strategy_name)
strategy=None
if strategy_name == 'fedavg':
  strategy = fl.server.strategy.FedAvg(initial_parameters=init_params, min_fit_clients=min_fl_clients_nb, min_available_clients=min_fl_clients_nb) 
elif strategy_name == 'fedadam':
  strategy = fl.server.strategy.FedAdam(initial_parameters=init_params, min_fit_clients=min_fl_clients_nb, min_available_clients=min_fl_clients_nb)
elif strategy_name == 'fedyogi':
  strategy = fl.server.strategy.FedYogi(initial_parameters=init_params, min_fit_clients=min_fl_clients_nb, min_available_clients=min_fl_clients_nb)
elif strategy_name == 'feddagrad':
  strategy = fl.server.strategy.FedAdagrad(initial_parameters=init_params, min_fit_clients=min_fl_clients_nb, min_available_clients=min_fl_clients_nb)
elif strategy_name == 'qffedavg':
  strategy = fl.server.strategy.QffedAvg(initial_parameters=init_params, min_fit_clients=min_fl_clients_nb, min_available_clients=min_fl_clients_nb)
else:
  raise ValueError('cannot recognize agregation strategy from the experiment settings file')

fl.server.start_server(config={"num_rounds": 50}, strategy=strategy)


