# ========================================
# FileName: start_kafka_producer.py
# Date: 29 june 2020 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of functionnalities to start a federated server
# for DeepLearningTools.
# =========================================
# main parameter server, should be started first

import numpy as np
import os
import importlib
from typing import Dict, Optional, Tuple
import flwr as fl
from flwr.common import NDArrays, Scalar
import json 
import deeplearningtools
from deeplearningtools import experiments_manager # make use of all the standard tools of the framework
from deeplearningtools import tools

import os

global monitored_value_threshold # monitored discrepancy value (loss or other) used to trigger logging on better model

def get_commands():
  """
  Defines the command line argument parser dedicated to running session.
  """
  # retreive command line arguents, all the standard commands 
  parser=tools.command_line_parser.get_commands()
  # add script specific arguments
  parser.add_argument("-w","--pretrainedmodelcheckpointpath", default="",
                      help="path to a previous model checkpoint that provides pretrained model weights")
  parser.add_argument("-rounds","--num_rounds", default=20, type=int,
                      help="set the maximum clients/server rounds number, defaults is 20")
  parser.add_argument("-agr","--agregation", default='', type=str,
                      help="override server agregation method, string is case sensitive, must target one of the FLower lib methods or one within this library, or leave empty to use the one specified in the experiment setting file")
  parser.add_argument("-minCl","--min_available_clients", default=-1, type=int,
                      help="override the minimum number of available clients to allow for a federated session run, default<0 such that the value specified in the settings file is used, set >0 to use your own spec.")
  parser.add_argument("-minFit","--min_fit_clients", default=-1, type=int,
                      help="override the minimum number of available clients to allow for a single federated round run, default<0 such that the value specified in the settings file is used, set >0 to use your own spec.")
  parser.add_argument("-minEval","--min_eval_clients", default=-1, type=int,
                      help="override the minimum number of clients required to run a single evaluation process, default<0 such that the value specified in the settings file is used, set >0 to use your own spec.")
  parser.add_argument("-locEpoch","--local_epochs", default=1, type=int,
                      help="override the number of epochs each client perform required to run a single round, default=0 such that the value replaces the one specified in the settings file generally used for centralized learning.")
  parser.add_argument("-sim","--simulation", action='store_true',
                      help="run in simulation mode i.e. all processes are conducted on the same shared machine")
  parser.add_argument("-log","--log_dir", default='', type=str,
                    help="path to store ray logs")

  #get framework and local command line arguments
  return parser

def get_custom_strategy(strategy_name):
  """
  Retrieve a custom strategy class based on the strategy_name.

  :param strategy_name: Name of the strategy.
  :type strategy_name: str

  :return: The custom strategy class.
  :rtype: str

  :raises ValueError: If the strategy fails to load.
  """
  strategy_cl=None
  print('Trying to load stategy from standard Flower strategies:', strategy_name)
  try:
    strategy_module = getattr(fl.server.strategy, strategy_name.lower())
    strategy_cl = getattr(strategy_module, strategy_name)
    print('Found ', strategy_cl)
  except Exception as e:
    print('Chosen strategy is not part of fl.server.strategy => Attempting to load custom strategy from deeplearningtools.helpers.federated.', strategy_name, 'Error:', e)
    try:
      strategy_module = importlib.import_module('deeplearningtools.helpers.federated.'+strategy_name.lower())
      strategy_cl = getattr(strategy_module, strategy_name)      
    except Exception as e:
      raise ValueError('start_federated_server:strategy_cl: Failed to load strategy {s}, error message={err}'.format(s=strategy_name, err=e))
    
  return strategy_cl

def get_evaluate_fn(usersettings, model, train_data, val_data, file_writer, log_dir):
    """
    Return evaluation metrics results for server-side evaluation.

    Reference: inspired from https://flower.dev/docs/evaluation.html

    :param usersettings: User settings for the evaluation function.
    :type usersettings: object
    :param model: The model to be evaluated.
    :type model: keras.Model
    :param val_data: Validation data for evaluation.
    :type val_data: dict
    :param file_writer: File writer for logging into tensorBoard.
    :type file_writer: object tf.summary.FileWriter
    :param log_dir: Directory for logging.
    :type log_dir: str

    :return: The evaluation metrics result for each out.
    :rtype: tuple
    """
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        global monitored_value_threshold
        # prepare model on the server side
        model.set_weights(parameters)  # Update model with the latest parameters
        learning_rate=usersettings.get_learningRate()
        loss=usersettings.get_total_loss(model)
        optimizer=usersettings.get_optimizer(model, loss, learning_rate)
        metrics=usersettings.get_metrics(model, loss)
        model.compile(optimizer=optimizer,
              loss=loss,# you can use a different loss on each output by passing a dictionary or a list of losses
              loss_weights=None,
              metrics=metrics) #can specify per output metrics : metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}

        #recover fresh callbacks
        all_callbacks_dict=deeplearningtools.tools.callbacks.define_callbacks(usersettings=usersettings,
                                                                          model=model,
                                                                          val_data=val_data,
                                                                          train_iterations_per_epoch=0,#not useful in that use case
                                                                          file_writer=file_writer,
                                                                          log_dir=log_dir,
                                                                          metrics=metrics,
                                                                          previous_model_params=None,
                                                                          custom_callbacks={},
                                                                          initial_value_threshold=monitored_value_threshold)
        with file_writer.as_default():
          result = model.fit(
                              x=train_data['data_pipeline'],
                              y=None,#train_data,
                              batch_size=None,
                              epochs=usersettings.nbEpoch*(server_round+1),
                              verbose=1,
                              callbacks=all_callbacks_dict.values(),
                              validation_split=0.0,
                              validation_data=val_data['data_pipeline'], #=> done at the evaluate method level
                              shuffle=True,
                              class_weight=None,
                              sample_weight=None,
                              initial_epoch=server_round*usersettings.nbEpoch,
                              steps_per_epoch=1,#TODO, OPTIONAL: train_data['steps_per_epoch'],
                              validation_steps=val_data['steps_per_epoch'],
                              validation_freq=1,
                              )#return_dict=True,)
          # try to trigger on_epoch_end for all metrics
          # -> this may required for some of them to finish their processing (ex: helpers.metrics::ConfusionMatrix draw the confusion matrix on the tensorboard logs)
          '''for callback in all_callbacks_dict.values():
            if hasattr(callback, 'on_epoch_end'):
              callback.on_epoch_end(epoch=server_round)
          '''
        reported_loss=usersettings.monitored_loss_name
        print('******************* result', result.history, 'reported_loss', reported_loss)
        
        return result.history[reported_loss], result.history

    return evaluate

def run(FLAGS, parameters={}):
  """
  Run the federated learning server.

  :param FLAGS: Command line arguments and flags.
  :type FLAGS: object

  :return: The result of the server run.
  :rtype: object

  """
  global monitored_value_threshold
  monitored_value_threshold=np.inf # indicator that triggers model checkpointing initially set to maximum value 
  #get experiment settings filename path
  usersettings_file=FLAGS.usersettings
  #load experiments settings, add eventual command line parameters andforce server mode to only prepare a session folder from which the server can report its states
  param_addons={'isFLserver':True}
  if len(FLAGS.agregation)>0:
    param_addons['federated']=FLAGS.agregation
  if FLAGS.min_available_clients>0:
    param_addons['minCl']=FLAGS.min_available_clients
  if FLAGS.min_fit_clients>0:
    param_addons['minFit']=FLAGS.min_fit_clients
  if FLAGS.min_eval_clients>0:
    param_addons['minEval']=FLAGS.min_eval_clients
  if FLAGS.local_epochs>0:
    param_addons['nbEpoch']=FLAGS.local_epochs
  if 'session_number' in parameters.keys():
    param_addons["session_number"] = parameters["session_number"]
  if 'session_folder' in parameters.keys():
    param_addons["session_folder"] = parameters["session_folder"]


  job_session_folder = experiments_manager.run(FLAGS, train_config_script=usersettings_file, external_hparams=param_addons)
  print('job_session_folder', job_session_folder)
  # Next keep initial working folder, move to the session folder and run experiment such that :
  # -> the current settings file can be loaded is necessary
  # -> any file writen in the process in os.getcwd() is kept in the experiment folder
  initial_wd=os.getcwd()
  os.chdir(job_session_folder)

  #load a model ready for training and more especially evaluation on the server side

  usersettings, model, train_data, val_data, file_writer = experiments_manager.build_run_training_session()

  #check GPU requirements vs availability
  num_available_GPU=0
  if len(usersettings.used_gpu_IDs)>0:
    available_GPUs=tools.gpu.get_available_gpus()
    print('Available GPUs:', available_GPUs)
    num_available_GPU=min(len(usersettings.used_gpu_IDs), len(available_GPUs))
  else:
    print('No GPU required for this experiment (usersettings.used_gpu_IDs is empty)')
  if num_available_GPU<len(usersettings.used_gpu_IDs):
    raise ValueError('Not enough GPU available for the federated server, check the used_gpu_IDs parameter in the experiment settings file')

  strategy_name=None
  if 'federated' in usersettings.hparams:
    strategy_name=usersettings.hparams['federated']
  else:
    raise ValueError('Experiment settings does not specify the federated agregation method (hparams[\'federated\']')

  if 'minCl' in usersettings.hparams:
    min_available_clients=usersettings.hparams['minCl']
  else:
    raise ValueError('Experiment settings does not specify the minimum number of clients (hparams[\'minCl\']')

  if 'minFit' in usersettings.hparams:
    min_fit_clients=usersettings.hparams['minFit']
    if min_fit_clients>min_available_clients:
      raise ValueError('min_fit_clients must be lower than or equal to min_available_clients, check parameters')
  else:
    raise ValueError('Experiment settings does not specify the minimum number of fitted clients (hparams[\'minFit\']')

  if 'minEval' in usersettings.hparams:
    min_eval_clients=usersettings.hparams['minEval']
    if min_eval_clients>min_available_clients:
      raise ValueError('min_eval_clients must be lower than or equal to min_available_clients, check parameters')
  else:
    min_eval_clients=min_fit_clients
    print('Experiment settings does not specify the minimum number of evaluated clients (hparams[\'minEval\'] -> automatically set to the min_fit_clients value')
  
  if os.path.exists(FLAGS.pretrainedmodelcheckpointpath):
    print('centralized model is pretrained... loading from ', FLAGS.pretrainedmodelcheckpointpath)
    from helpers.model import load_model
    model_pretrained=load_model(os.path.join(FLAGS.pretrainedmodelcheckpointpath,'checkpoints/'), usersettings)
    model.set_weights(model_pretrained.get_weights())

  #load model as will be done by the clients taking into account the experiment settings
  #model, callbacks = experiments_manager.build_run_training_session()
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

  #prepare the evaluation function on the server side
  eval_fn=get_evaluate_fn(usersettings, model, train_data, val_data, file_writer, os.path.join(os.getcwd(),"logs"))

  #then apply the specified agregation strategy
  print('Experiment settings file reports federated learning strategy:', strategy_name)
  strategy_cl = get_custom_strategy(strategy_name)
  strategy=strategy_cl(initial_parameters=init_params, min_fit_clients=min_fit_clients, min_evaluate_clients=min_eval_clients, min_available_clients=min_available_clients, evaluate_fn=eval_fn) 
  
  #finally run server in real or simulation mode
  if FLAGS.simulation is False:
    print('Federated Learning session starts with REAL clients, CHECK DISTRIBUTED RESSOURCES availability and load')
    result=fl.server.start_server(server_address="localhost:8080", config=fl.server.ServerConfig(num_rounds=FLAGS.num_rounds), strategy=strategy)
  else:
    print('Federated Learning session starts with SIMULATED clients, SINGLE MACHINE MODE, check machine load')
    print('--> have a look at logs at /tmp/ray/session_latest/logs/')

    #UGLY IMPORT but ray should be loaded only once here in case this script is called multiple times
    if 'ray' not in dir():
      import ray
      if parameters is None:
        ray.init(num_gpus=num_available_GPU) # @bug might be a problem for a multi experiments run


    # @debug
    #ressources = ray.available_resources() # ray will be imported by flower in the next few lines
    #print('INFO : available ressources reported by ray: ', ressources)

    client_training_fn=experiments_manager.build_run_training_session
    # before starting simulation, help local modules to be loaded properly
    # on the ray client side by adding them a __path__ property to mke them loadable  
    deeplearningtools.__path__=[os.path.join(initial_wd, 'deeplearningtools')]
    
    if 'tmp_dir' not in parameters.keys():
      parameters['tmp_dir'] = "/tmp/ray/"
    #setup ray server config:
    ray_server_config={
              "ignore_reinit_error": True, #default arg
              "include_dashboard": True, # default is False but you may need this for tracking
              #"_temp_dir":job_session_folder+"/ray",
              "num_cpus": 6,
              "num_gpus": num_available_GPU,
              "_temp_dir": parameters['tmp_dir'],
              #"_memory": 16000 * 1024 * 1024,#16Gb
              "runtime_env":{ "working_dir":os.path.join(initial_wd,job_session_folder),
                              "py_modules": [deeplearningtools],                          
                            }
    }
    # FIXME, maybe use below 'client_resources' or delete
    client_resources = {
            "num_cpus": 2,
            "num_gpus": 1.0/(usersettings.hparams['minCl']+1) if num_available_GPU >0 else 0,# 0.2,
    }

    # start simulation           
    result=fl.simulation.start_simulation(
      client_fn=client_training_fn,
      num_clients=min_available_clients,
      config=fl.server.ServerConfig(num_rounds=FLAGS.num_rounds),
      strategy=strategy,
      ray_init_args=ray_server_config,
      client_resources=client_resources,
    )
    # create a symbolic link to the ray logs related to simulated clients into the main experiment folder
    os.symlink(os.path.realpath("/tmp/ray/session_latest/"), os.path.join(os.getcwd(), 'ray_logs'))
    
  log_path = "metrics"
  filename = "metrics.json"
  metrics = {}
  metrics["centralized"] = result.metrics_centralized
  metrics["distributed"] = result.metrics_distributed

  path = os.path.join(os.getcwd(),log_path)

  os.makedirs(path, exist_ok=True)
  print(os.path.join(os.getcwd(),log_path, filename))
  with open(os.path.join(os.getcwd(),log_path, filename), 'w') as outfile:
    outfile.write(json.dumps(metrics, indent = 4) )


  #end of the job, recover initial working directory
  os.chdir(initial_wd)
  print("###################### FEDERATED SERVER END ######################")
  print("Result:", result, type(result))
  return result

if __name__ == "__main__":
  # Make TensorFlow logs less verbose
  #os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

  # retreive command line arguents
  parser = get_commands()
  FLAGS=parser.parse_args()
  
  run(FLAGS)
