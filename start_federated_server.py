# main parameter server, should be started first
import flwr as fl
import tensorflow as tf
import os
import experiments_manager # make use of all the standard tools of the framework
import tools
import helpers
import sys

sys.path.insert(0, os.getcwd())
# retreive command line arguents, all the standard commands 
parser=tools.command_line_parser.get_commands()
# add script specific arguments
parser.add_argument("-w","--pretrainedmodelcheckpointpath", default="",
                    help="path to a previous model checkpoint that provides pretrained model weights")
parser.add_argument("-rounds","--num_rounds", default=20, type=int,
                    help="set the maximum clients/server rounds number, defaults is 100")
parser.add_argument("-clients","--num_clients", default=5, type=int,
                    help="set the maximum clients/server rounds number, defaults is 100")
parser.add_argument("-sim","--simulation", action='store_true',
                    help="run in simulation mode i.e. all processes are conducted on the same shared machine")
#get framework and local command line arguments
FLAGS = parser.parse_args()

#get experiment settings filename path
usersettings_file=FLAGS.usersettings
#load experiments settings, force server mode to only prepare a session folder from which the server can report its states
job_session_folder = experiments_manager.run(FLAGS, train_config_script=usersettings_file, external_hparams={'isFLserver':True})
print('job_session_folder', job_session_folder)
# Next keep initial working folder, move to the session folder and run experiment
initial_wd=os.getcwd()

usersettings, _= experiments_manager.loadExperimentsSettings(filename=os.path.join(job_session_folder, experiments_manager.SETTINGSFILE_COPY_NAME), 
                                                             call_from_session_folder=False)
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
  model=load_model(os.path.join(FLAGS.pretrainedmodelcheckpointpath,'checkpoints/'), usersettings)
else:
  print('Loading a clean model to be trained from scratch')
  model=experiments_manager.loadModel_def_file(usersettings, absolute_path=True)(usersettings)

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
init_params=fl.common.ndarrays_to_parameters(model.get_weights())

#then apply the specified agregation strategy
print('Experiment settings file reports federated learning strategy:', strategy_name)
strategy_cl=None
if strategy_name == 'fedavg':
  strategy_cl = fl.server.strategy.FedAvg

elif strategy_name == 'fedadam':
  strategy_cl = fl.server.strategy.FedAdam
elif strategy_name == 'fedyogi':
  strategy_cl = fl.server.strategy.FedYogi
elif strategy_name == 'feddagrad':
  strategy_cl = fl.server.strategy.FedAdagrad
elif strategy_name == 'qffedavg':
  strategy_cl = fl.server.strategy.QffedAvg
else:
  raise ValueError('cannot recognize agregation strategy from the experiment settings file')
strategy=strategy_cl(initial_parameters=init_params, min_fit_clients=min_fl_clients_nb, min_available_clients=min_fl_clients_nb) 
#finally run server in real or simulation mode
if FLAGS.simulation==False:
  print('Federated Learning session starts with REAL clients, CHECK DISTRIBUTED RESSOURCES availability and load')
  fl.server.start_server(server_address="localhost:8080", config=fl.server.ServerConfig(num_rounds=FLAGS.num_rounds), strategy=strategy)
else:
  print('Federated Learning session starts with SIMULATED clients, SINGLE MACHINE MODE, check machine load')
  print('--> have a look at logs at /tmp/ray/session_latest/logs/')
  import ray
  ray.init()
  ray.available_resources()
  client_training_fn=experiments_manager.build_run_training_session
  # before starting simulation, help local modules to be loaded properly
  # on the ray client side by adding them a __path__ property to mke them loadable  
  experiments_manager.__path__=[os.getcwd()]
  tools.__path__=[os.getcwd()]
  helpers.__path__=[os.getcwd()]

  #setup ray server config:
  ray_server_config={
            "ignore_reinit_error": True, #default arg
            "include_dashboard": True, # default is False but you may need this for tracking
            "num_cpus": 8,
            "num_gpus": 1,
            #"memory": 16000 * 1024 * 1024,#16Gb
            "runtime_env":{ "working_dir":job_session_folder,
                            "py_modules": [experiments_manager, tools, helpers]}
  }
    # The argument below is new
  client_resources = {
          "num_cpus": 2,
          "num_gpus": 0.2,
  }

  # start simulation           
  fl.simulation.start_simulation(
    client_fn=client_training_fn,
    num_clients=FLAGS.num_clients,
    config=fl.server.ServerConfig(num_rounds=FLAGS.num_rounds),
    strategy=strategy,
    ray_init_args=ray_server_config
  )

#end of the job, recover initial working directory
os.chdir(initial_wd)

