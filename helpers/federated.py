# a set of helpers facilitating the implementation of federated learning with the flower library
import tensorflow as tf
import flwr as fl
import numpy as np
from tools.experiment_settings import define_callbacks

# Define Flower client
class FlClient(fl.client.NumPyClient):
    def __init__(self, settings, model, train_data, train_iterations_per_epoch, val_data, val_iterations_per_epoch, workers, file_writer, log_dir):
        self.history=None
        self.round=0
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
        self.all_callbacks_dict=define_callbacks(self.settings, self.model, self.train_iterations_per_epoch, self.file_writer, self.log_dir)

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print('#################### FlClient.fit new round', self.round)
        #set updated weights
        self.model.set_weights(parameters)
        # (re)define each callbacks
        if self.round>0:
            self.all_callbacks_dict=define_callbacks(self.settings, self.model, self.train_iterations_per_epoch, self.file_writer, self.log_dir, previous_model_params=self.model.get_weights())

        #training for one epoch
        history=self.model.fit(
            x=self.train_data,
            y=None,#train_data,
            batch_size=None,
            epochs=self.round+1,
            verbose=1,
            callbacks=self.all_callbacks_dict.values(),
            validation_split=0.0,
            validation_data=self.val_data, #=> done at the evaluate method level
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=self.round,
            steps_per_epoch=self.train_iterations_per_epoch,
            validation_steps=self.val_iterations_per_epoch,
            validation_freq=1,
            max_queue_size=self.workers*100,
            workers=self.workers,
            use_multiprocessing=False,
        )
        
        self.model.track_weights_change(parameters, self.round)
        print('FlClient.fit round result, history=', self.round,history.history)
        logs_last_epoch={ key:history.history[key][-1] for key in history.history.keys()}
        print('==> last history=', logs_last_epoch)

        if len(history.history)>0:
            self.history=history
        # avoiding to reuse callbacks : only affect them on the first round
        self.round+=1

        return self.model.get_weights(), self.train_iterations_per_epoch*self.settings.batch_size, logs_last_epoch#{}

    def evaluate(self, parameters, config):
        print('Evaluating model from received parameters...')
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
        print('FlClient.evaluate losses:',losses)

        return losses[self.settings.monitored_loss_name], self.val_iterations_per_epoch*self.settings.batch_size, losses#self.history.history#loss_dict

# a generic Flower client class, from https://github.com/adap/flower/blob/main/examples/simulation_tensorflow/sim.ipynb
class FlowerClient_(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_val, y_val) -> None:
        ''' generic constructor that merges a model and data sources
          allocated externally into a single Flower client'''
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, verbose=2)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}



