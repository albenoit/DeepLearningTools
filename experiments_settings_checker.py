'''
@author : Alexandre Benoit, LISTIC lab, FRANCE
@brief  : a set of tools to validate the experiments settings file used to train and serve a given model
'''

import os
class ExperimentsSettingsChecker(object):

    def assertPositive_above_zero(self, param, param_description):
            message='Specification error on variable {param}: {descr}. It must be set and be above 0'.format(param=param, descr=param_description)
            assert hasattr(self.experiments_settings,param), message
            assert getattr(self.experiments_settings,param)>0, message

    def has(self, param, error_message):
        assert hasattr(self.experiments_settings, param), 'Missing {param} : {msg}'.format(param=param, msg=error_message)

    def __init__(self,experiments_settings):
        self.experiments_settings=experiments_settings

    def validate_settings(self):
            print('Checking the experiments settings file...')
            print('** look at the README.md file to read a working example')
            print('** look at the experiments_settings_checker script to see all the required fields')
            #check model
            self.has('model_file', 'model_file must be set as a filename targetting the model description to optimise')
            assert os.path.exists(self.experiments_settings.model_file), '{model} targetted by model_file filename does not exist'.format(experiments_settings.model_file)
            self.assertPositive_above_zero('patchSize', 'the extend (in pixels/data samples width) of the input data samples provided to the model')
            self.assertPositive_above_zero('field_of_view', 'the width/field of view of the model. With convolutionnal models, this corresponds to the neighborhood width in the input space that is taken into account to take a decision')

            #check train and validation dataset parameters
            self.assertPositive_above_zero('nb_train_samples', 'the number of samples used for training')
            self.assertPositive_above_zero('nb_test_samples', 'the number of samples used for validation')

            #check standard training parameters
            self.assertPositive_above_zero('batch_size', 'the number of samples processed for each batch')
            self.assertPositive_above_zero('nbEpoch', 'the number of times the training set is processed for training')
            self.assertPositive_above_zero('initial_learning_rate', 'the training speed factor')
            self.assertPositive_above_zero('nb_classes', 'the number of classes for classification problems of the code size for embedding problems')
            self.has('num_epochs_per_decay', 'integer value that, ONLY IF ABOVE 0 specifies after how many training epoch one must apply a decay to the learning rate')
            self.has('learning_rate_decay_factor', 'float value factor to be applied to the learning rate when decaying is applied')
            self.has('used_gpu_IDs', 'a list of integer(s) ID(s) that specify which GPU to use relying on their ID')
            self.has('workingFolder', 'a string that specifies the parent pathname where the training procedure data is being stored')

            #train and validation flags and functions
            self.has('train_val_schedule_strategy', 'the strategy for training and validation cycles, according to Tensorflow doc, set  \'continuous_train_and_eval\' when running on a single GPU or \'train_and_evaluate\' when multiple machines/GPUs are available simultaneously ')
            self.has('predict_using_smoothed_parameters', 'a boolean that if True (not yet working, prefer False) applies exponential moving average updates on the saved models used for validation and prediction.')
            self.has('get_total_loss', 'function that receives parameters [inputs, model_outputs_dict, labels, weights_loss] that must return a graph node that represents the optimisation loss. This node will be drawn on the tensorboard as the \'loss\' variable.')
            self.has('get_eval_metric_ops', 'function that receives parameters [inputs, model_outputs_dict, labels] and that returns a dictionnary of tf.metrics')
            self.has('getOptimizer', 'function that receives as parameters [loss, learning_rate, global_step] and that outputs an optimisation node (generally a loss minimization op)')
            self.has('reference_labels', 'a list of strings THAT MUST BE of the same length as the number or model outputs')

            #input pipelines
            self.has('get_input_pipeline_train_val', 'the train and validation input data pipelines function params=[batch_size, raw_data_files_folder, shuffle_batches], must return an input function as described here : https://www.tensorflow.org/programmers_guide/datasets')
            self.has('get_input_pipeline_serving', 'the input pilepeline to be used when serving the model with tensorflow_server, must return a tf.estimator.export.ServingInputReceiver instance')

            #data pre and post processing
            self.has('data_preprocess', 'function that receives parameters [features, model_placement] and outputs a graph node that applies some preprocessing on \'features\' on the desired device knowing the fact that the model will be placed on device \'model_placement\'. Such processing is applied to the input data whatever the mode: training, validation or prediction')
            self.has('model_outputs_postprocessing_for_serving', 'function that receives the model outputs dictionnary and that returns a post processed version')

            #degug and log flags
            self.has('display_model_layers_info', 'boolean to activate or not detailled layers display when constructing the model')
            self.has('nb_summary_per_train_epoch', 'integer nb_summary_per_train_epoch must be set to specify monitored values logging period along training. value<0 forces to log at each training step')

            #tensorflow serving and client dialog
            self.assertPositive_above_zero('wait_for_server_ready_int_secs', 'the number of seconds to wait for a tensorflow service before timeout on first contact')
            self.assertPositive_above_zero('serving_client_timeout_int_secs', 'the number of seconds to wait for a tensorflow service before timeout for each request')

            self.has('Client_IO', 'a Client_IO class that defines how a client talks to a tensorflow server')
            self.has('served_head', 'string output that will be provided by tensorflow-server')
            self.has('input_data_name', 'string that specifies the node name of the input data pipeline used when serving a model and that is used by the tensorflow-server')
            self.has('tensorflow_server_address', 'a string specifying the IP adress of the tensorflow server to be contacted by a client')
            self.has('tensorflow_server_port', 'an integer that specifies the port use to communicate whith the tensorflow server')

            print('...No problem detected at this step.')
