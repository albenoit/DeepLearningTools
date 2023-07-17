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
import os, importlib
from typing import Dict, Optional, Tuple
import flwr as fl
from flwr.common import NDArrays, Scalar
import tensorflow as tf
import deeplearningtools
from deeplearningtools import experiments_manager # make use of all the standard tools of the framework
from deeplearningtools import tools
from deeplearningtools.tools.flower_utils import plot_metric_from_history

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
    print('Chosen strategy is not part of fl.server.strategy => Attempting to load custom strategy from deeplearningtools.helpers.federated.', strategy_name)
    try:
      strategy_module = importlib.import_module('deeplearningtools.helpers.federated.'+strategy_name.lower())
      strategy_cl = getattr(strategy_module, strategy_name)      
    except Exception as e:
      raise ValueError('start_federated_server:strategy_cl: Failed to load strategy {s}, error message={err}'.format(s=strategy_name, err=e))
    
  return strategy_cl

def get_evaluate_fn(usersettings, model, val_data, file_writer, log_dir):
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
        #recover fresh callbacks
        all_callbacks_dict=deeplearningtools.tools.callbacks.define_callbacks(usersettings=usersettings,
                                                                          model=model,
                                                                          train_iterations_per_epoch=0,#not useful in that use case
                                                                          file_writer=file_writer,
                                                                          log_dir=log_dir,
                                                                          metrics=usersettings.get_metrics(model, None),
                                                                          previous_model_params=None,
                                                                          custom_callbacks={},
                                                                          initial_value_threshold=monitored_value_threshold)
        model.set_weights(parameters)  # Update model with the latest parameters
        res_raw = model.evaluate(val_data['data_pipeline'],
                                       steps=val_data['steps_per_epoch'],
                                       callbacks=all_callbacks_dict.values())
        # try to trigger on_epoch_end for all metrics
        # -> this may required for some of them to finish their processing (ex: helpers.metrics::ConfusionMatrix draw the confusion matrix on the tensorboard logs)
        for metric in model.metrics:
          if hasattr(metric, 'on_epoch_end'):
            metric.on_epoch_end(epoch=round)
        #workaround related to https://github.com/keras-team/keras/issues/14045
        print('reformating evaluation results, expecting metrics:', model.metrics_names)
        result = {out: res_raw[i] for i, out in enumerate(model.metrics_names)}
        print('metrics', result)
        return result['loss'], result

    return evaluate

def run(FLAGS):
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

  job_session_folder = experiments_manager.run(FLAGS, train_config_script=usersettings_file, external_hparams=param_addons)
  print('job_session_folder', job_session_folder)
  # Next keep initial working folder, move to the session folder and run experiment such that :
  # -> the current settings file can be loaded is necessary
  # -> any file writen in the process in os.getcwd() is kept in the experiment folder
  initial_wd=os.getcwd()
  os.chdir(job_session_folder)
  # path where flower results synthesis will be saved
  flower_report_path=f"flower_results"
  os.makedirs(flower_report_path, exist_ok=True)
  #load a model ready for training and more especially evaluation on the server side
  usersettings, model, train_data, val_data, file_writer = experiments_manager.build_run_training_session()
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
  eval_fn=get_evaluate_fn(usersettings, model, val_data, file_writer, os.path.join(os.getcwd(),"logs"))

  #then apply the specified agregation strategy
  print('Experiment settings file reports federated learning strategy:', strategy_name)
  strategy_cl = get_custom_strategy(strategy_name)
  strategy=strategy_cl(initial_parameters=init_params, min_fit_clients=min_fit_clients, min_evaluate_clients=min_eval_clients, min_available_clients=min_available_clients, evaluate_fn=eval_fn) 
  

  #finally run server in real or simulation mode
  if FLAGS.simulation==False:
    print('Federated Learning session starts with REAL clients, CHECK DISTRIBUTED RESSOURCES availability and load')
    result=fl.server.start_server(server_address="localhost:8080", config=fl.server.ServerConfig(num_rounds=FLAGS.num_rounds), strategy=strategy)
  else:
    print('Federated Learning session starts with SIMULATED clients, SINGLE MACHINE MODE, check machine load')
    print('--> have a look at logs at /tmp/ray/session_latest/logs/')
    import ray
    ray.init()
    ressources = ray.available_resources()
    print('INFO : available ressources reported by ray: ', ressources)
    client_training_fn=experiments_manager.build_run_training_session
    # before starting simulation, help local modules to be loaded properly
    # on the ray client side by adding them a __path__ property to mke them loadable  
    deeplearningtools.__path__=[os.path.join(initial_wd, 'deeplearningtools')]
    
    #setup ray server config:
    ray_server_config={
              "ignore_reinit_error": True, #default arg
              "include_dashboard": True, # default is False but you may need this for tracking
              #"_temp_dir":job_session_folder+"/ray",
              "num_cpus": 6,
              "num_gpus": 1,
              #"_memory": 16000 * 1024 * 1024,#16Gb
              "runtime_env":{ "working_dir":os.path.join(initial_wd,job_session_folder),
                              "py_modules": [deeplearningtools],                          
                            }
    }
      # The argument below is new
    client_resources = {
            "num_cpus": 2,
            "num_gpus": 0.2,
    }

    # start simulation           
    result=fl.simulation.start_simulation(
      client_fn=client_training_fn,
      num_clients=min_available_clients,
      config=fl.server.ServerConfig(num_rounds=FLAGS.num_rounds),
      strategy=strategy,
      ray_init_args=ray_server_config
    )
    os.symlink(os.path.realpath("/tmp/ray/session_latest/"), os.path.join(os.getcwd(), 'ray_logs'))

  #finally build and save Flwr results synthesis:
  np.save(
        flower_report_path+f'/results.npy',
        result,  # type: ignore
    )

  plot_metric_from_history(
        result,
        flower_report_path,
        "flower_report",
    )

  #end of the job, recover initial working directory
  os.chdir(initial_wd)
  return result

if __name__ == "__main__":
  # Make TensorFlow logs less verbose
  #os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

  # retreive command line arguents
  parser = get_commands()
  FLAGS=parser.parse_args()
  run(FLAGS)
