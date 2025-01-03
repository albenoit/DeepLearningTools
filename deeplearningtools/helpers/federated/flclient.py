# ========================================
# FileName: flclient.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of helpers facilitating the implementation of federated learning with the flower library
# for DeepLearningTools.
# =========================================

import os
import configparser
import flwr as fl
import numpy as np
from deeplearningtools.tools.callbacks import define_callbacks
from deeplearningtools.helpers.distance_metrics.metrics import METRIC_SET, ModelMetric

# -------------------------------------------
# A custom Flower client able to manage warm restart,
#  manage optimisation history comparable with centralized learning,
#  make use of experiment settings
#  and relevant callbacks management to save model checkpoints and logs
# -------------------------------------------

class FlClient(fl.client.NumPyClient):
    def __init__(self, settings, model, train_data, train_iterations_per_epoch, val_data, val_iterations_per_epoch, workers, file_writer, log_dir, metrics=[], monitored_metric_initial_threshold=None):
        """
        Constructor for the FlClient. It merges a model and data sources
        allocated externally into a single Flower client.
        Args:
            settings: Experiment settings that specify experiment hypermparameters, metrics, etc.
            model: Model object.
            train_data: Training data features.
            train_iterations_per_epoch: Number of training iterations per epoch.
            val_data: Validation data features.
            val_iterations_per_epoch: Number of validation iterations per epoch.
            workers: Number of workers.
            file_writer: File writer.
            log_dir: Log directory.
            metrics: List of metrics.
            monitored_metric_initial_threshold: Initial threshold for monitored metric (last best know loss value).
        """
        self.history=None
        self.round=-1
        self.last_val_loss=np.inf
        self.settings=settings
        self.model=model
        self.train_data = train_data
        self.train_iterations_per_epoch=train_iterations_per_epoch
        self.val_data = val_data
        self.val_iterations_per_epoch=val_iterations_per_epoch
        self.workers=workers
        self.file_writer=file_writer
        self.log_dir=log_dir
        self.metrics=metrics
        #initialized a first time to allow for model first evaluation
        self.initial_value_threshold=monitored_metric_initial_threshold
        self.all_callbacks_dict=define_callbacks(self.settings, self.model,self.val_data, self.train_iterations_per_epoch, self.file_writer, self.log_dir, metrics=self.metrics, previous_model_params=self.model.get_weights(), initial_value_threshold=self.initial_value_threshold)
        # some log info recovered on new round participation
        self.rounds_calls=0

        self.dst_metrics = METRIC_SET.get_metrics(["fb-trusted_dst", "fb-model_cosine_dst"]) 

    def get_client_config_filename(self):
        """
        Just a helper to ensure consistent warm restart config file naming.
        
        Returns a standardized config name.
        
        :return: Standardized config name.
        :rtype: str
        """
        file_id='lastinfo_client'+str(self.settings.hparams['procID'])+'.ini'
        file_fullpath=os.path.join(os.getcwd(), file_id)
        return file_fullpath
    
    def write_restart_config(self, monitored_loss_val):
        """
        Writes a config file (after each client.fit call) to save last states, enabling warm restart.
    
        This function saves all necessary data, including the last best monitored value, epoch index, etc.,
        in order to better track the client model.
        
        :param monitored_loss_val: The model's monitored metric.
        :type monitored_loss_val: float
        
        Writes a configuration file named 'lastinfo_clientXXX.ini' in the current working directory.
        """
        #prepare a config file for model serving
        config_filename=self.get_client_config_filename()
        client_config = configparser.ConfigParser()
        client_config['LASTSTATE'] = { 'rounds_participation': self.rounds_calls,
                                        'last_round_participation':self.round,
                                        'monitored_loss_value': monitored_loss_val,
                                    }
        with open(config_filename, 'w') as configfile:
          client_config.write(configfile)

    def load_restart_config(self):
        """
        Loads a config file (if it exists) to recover last states, enabling warm restart.
        
        This function loads all necessary data, including the last best monitored value, epoch index, etc.,
        in order to better track the client model.
        
        :param None:
        
        :return: Nothing
        
        Updates some instance variables.
        """
        config_filename=self.get_client_config_filename()
        print('### Trying to load restart config file: ', config_filename)
        try:
            client_config = configparser.ConfigParser()
            client_config.read(config_filename, encoding='utf8')
            self.rounds_calls=int(client_config['LASTSTATE']['rounds_participation'])
            self.round=int(client_config['LASTSTATE']['last_round_participation'])
            self.initial_value_threshold=float(client_config['LASTSTATE']['monitored_loss_value'])
            print('### -> recovered rounds participation count:', self.rounds_calls)
            print('### -> recovered last round participation:', self.round)
            print('### -> recovered last best monitored value:', self.initial_value_threshold)
        except Exception as e:
            print('### -> could not load client warmstart file, maybe this is the first call')

    def get_parameters(self):
        """
        Get the client model parameters.

        :return: Model parameters.
        """
        return self.model.get_weights()

    def fit(self, parameters, config):
        """
        Fit the client model.

        :param parameters: Model parameters.
        :param config: Configuration dictionary.
        :return: Tuple containing updated model parameters, number of training samples, and a dictionary of log metrics.
        """
        print('#################### FlClient.fit new round')
        #try to load warm start info
        self.load_restart_config()
        print('### => server sent config:', config)
        if 'server_round' in config.keys():
            self.round=config['server_round']
        else:
            print('### => config does not report server_round property, managing rounds on the client side')
            self.round+=1
        print('## round : ', self.round)
        self.rounds_calls+=1 #one more round participation
        
        # define each callbacks
        if self.round>0:
            self.all_callbacks_dict=define_callbacks(self.settings, self.model, self.val_data, self.train_iterations_per_epoch, self.file_writer, self.log_dir, metrics=self.metrics, previous_model_params=self.model.get_weights(), initial_value_threshold=self.initial_value_threshold)

        #set updated weights
        self.model.set_weights(parameters)

        #training for one epoch
        history=self.model.fit(
            x=self.train_data,
            y=None,#train_data,
            batch_size=None,
            epochs=self.settings.nbEpoch*(self.round+1),
            verbose=1,
            callbacks=self.all_callbacks_dict.values(),
            validation_split=0.0,
            validation_data=self.val_data, #=> done at the evaluate method level
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=self.round*self.settings.nbEpoch,
            steps_per_epoch=self.train_iterations_per_epoch,
            validation_steps=self.val_iterations_per_epoch,
            validation_freq=1,
            max_queue_size=self.workers*100,
            workers=self.workers,
            use_multiprocessing=False,
        )
        
        change_from_global_metrics=self.model.track_weights_change(parameters, self.round, prefix='round_change_from_global', dst_metrics=self.dst_metrics)
        print('Client fit step done')
        print('-> FlClient.fit round result, history=', self.round,history.history)

        if len(history.history)>0:
            self.history=history
        
        # prepare emitted logs
        fit_log = { key:history.history[key][-1] for key in history.history.keys()} #{k: np.mean(v) for k, v in history.history.items()}
        fit_log.update(change_from_global_metrics)

        fit_log['client_id'] = self.settings.hparams['procID']
        
        # TODO, maybe check earlier if some metrics return non compatible types
        # =>  scalars are expected to be python dtype, no numpy allowed in the current Flwer state
        for key in fit_log.keys():
            if isinstance(fit_log[key], np.floating):
                fit_log[key]=float(fit_log[key])
        print('==> fit log=', fit_log)

        #save last fit log values that will allow for updated restart
        #print('LAST MONITORED VALUE=', (self.settings.monitored_loss_name,
        #                               self.all_callbacks_dict['checkpoint_callback'].best))
        self.write_restart_config(self.all_callbacks_dict['checkpoint_callback'].best)

        #update self.initial_value_threshold wrt last monitored value
        #-> will trigger checkpointing/model export only if best results are obtained
        self.initial_value_threshold=fit_log[self.settings.monitored_loss_name]

        return self.model.get_weights(), int(self.train_iterations_per_epoch*self.settings.batch_size), fit_log

    def evaluate(self, parameters, config):
        """
        Evaluate the client model.

        :param parameters: Model parameters.
        :param config: Configuration dictionary.
        :return: Tuple containing loss, number of validation samples, and a dictionary with accuracy.
        """
        print('Evaluating model from received parameters...')
        print('config: ', config)
        self.model.set_weights(parameters)
        losses = self.model.evaluate(x=self.val_data,
                                y=None,
                                batch_size=None,
                                verbose=1,
                                sample_weight=None,
                                steps=self.val_iterations_per_epoch,
                                callbacks=self.all_callbacks_dict.values(),
                                max_queue_size=10,
                                workers=self.workers,
                                use_multiprocessing=False,
                                return_dict=True
                                )

        reported_loss=self.settings.monitored_loss_name
        if reported_loss not in losses.keys():
            #remove a possible 'val_' prefix
            reported_loss_noprefix=self.settings.monitored_loss_name[4:]
            if reported_loss_noprefix in losses.keys():
                reported_loss=reported_loss_noprefix
        print('FlClient.evaluate monitored losses:', reported_loss)

        return losses[reported_loss],int(self.val_iterations_per_epoch*self.settings.batch_size), losses#self.history.history#loss_dict

