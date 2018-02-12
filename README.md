#What's that ?
A set of script that demonstrate the use of Tensorflow experiments and estimators on 1D data
@brief : the main script that enables training, validation and serving Tensorflow based models merging all needs in a
single script to train, evaluate, export and serve.
taking large inspirations of official tensorflow demos.
@author : Alexandre Benoit, LISTIC lab, FRANCE

Several ideas are put together:
-experiments and estimators to manage training, valiation and export in a easier way (but experiments are still in the contrib module so subject to strong changes)
-using moving averages to store parameters with values smoothed along the last training steps (FIXME : ensure those values are used for real by the estimator, actually the graph shows 2 parameter savers...).
-visualization including embeddings projections to observe some data projections on the TensorBoard
-tensorflow-serving api use to serve the model and dynamically load updated models
-some tensorflow-serving client codes to reuse the trained model on single or streaming data


#Machine Setup (tested with tensorflow 1.4.1)
1. install python 2.7 and python pip
2. install tensorflow and tensorflow serving using pip : pip install tensorflow-gpu tensorflow-serving-api
Note that the first versions of the dependency lib grpcio may bring some troubles when starting the tensorflow server.
grpcio python library version 1.7.3 and latest version above 1.8.4 should work.
==> Additionnal recommendations:
Get much better performances with optimized tensorflow packages coming from here:
https://github.com/mind/wheels/releases/
Install like this adajust the last link to your target version:
pip install --ignore-installed --upgrade \ https://github.com/mind/wheels/releases/download/tf1.4.1-gpu-cuda9/tensorflow-1.4.1-cp27-cp27mu-linux_x86_64.whl
Get the Intel MKL library installed :
https://github.com/mind/wheels#mkl


#Demo with a pretrained network
A pretrained autoencoder network working on time series is provided with the codes to see what you can get :
1. open a terminal and go into the source code directory
2. start the tensorboard on the experiments/1Dsignals_clustering folder using commands
--> tensorboard --logdir=experiments/1Dsignals_clustering
3. open a web browser and get to http://127.0.0.1:6006/ and observe:
-->on the "Scalars" section the evolution of the train and test loss, Mean Squared Error (MSE) and some other monitors evolution measured along a training and validation session
-->go to the "Graphs" section to observe the network training and testing architectures
-->go to the "Distributions" and "Histograms" sections to observe the evolution of the learned parameters along training
-->go to the "Projector" section and choose on the left pane a saved tensor (input data, embedding code or reconstructed data) to project them using PCA or t-SNE and interact with the interface (select points, etc.) to observe the projected time series blocks and their neighborhood.
4. serve and interact with this trained model:
--> start the tensorflow-server on this model using command:
python experiments_manager.py --start_server --model_dir=experiments/1Dsignals_clustering/my_test_2018--05--11\:34\:05/
--> start a client and ask to encode an input signal using command:
python experiments_manager.py --predict --model_dir=experiments/1Dsignals_clustering/my_test_2018-01-05--11\:34\:05/

#How tu use it to train/test/serve a new model for a new use case ?

The main script is experiments_manager.py can be used in 3 modes, here are some command examples:
1. train a model in a context specified in a parameters script such as mysettings_1D_experiments.py (details provided in the following TODO section):
-> python experiments_manager.py --usersettings=mysettings_1D_experiments.py
2. start a tensorflow server on the trained/training model :
-> python experiments_manager.py --start_server --model_dir=experiments/1Dsignals_clustering/my_test_2018-01-03--14:40:53
3. interract with the tensorflow server, sending input buffers and receiving answers
-> python experiments_manager.py --predict --model_dir=experiments/1Dsignals_clustering/my_test_2018-01-03--14\:40\:53/

NOTE : once trained (or along training), start the Tensorboard to parse logs of
the experiments folder (provided example is experiments/1Dsignals_clustering):
from the scripts directory using command: tensorboard  --logdir=experiments/1Dsignals_clustering
Then, open a web brwser and reach http://127.0.0.1:6006/ to monitor training
values and observe the obtained embeddings

#DESIGN:

1. The main code for training, validation and prediction is specified in the main script (experiments_manager.py).
2. Most of the use case specific parameters and Input/Output functions have been
moved to a separated settings script such as 'mysettings_1D_experiments.py' that
is targeted when starting the script (this filename is set in var FLAGS.usersettings in the main script).
3. The model to be trained and served is specified in a different script targeted by the settings file.

#KNOWN ISSUES :

This script has some known problems, any suggestion is welcome:
-moving average parameters saving is maybe not correctly done. I am not sure that the smoothed variables are saved instead of the current parameters
-for now tensorflow_server only works on CPU so using GPU only for training and validation. Track : https://github.com/tensorflow/serving/issues/668

#TODO :

To adapt to new use case, just update the mysettingsxxx.py file and adjust I/O functions.
For any experiment, the availability of all the required fields in the settings file is checked by the experiments_settings_checker.py script. You can have a look there to ensure you prepared everything right.
In addition and as a reminder, here are the functions prototypes:
-define a model to be trained and served in a specific file and follow this prototype:
--report model name in the settings file
--def model( data, #the input data tensor
            n_outputs, #dimension of the embedding code
            hparams,  #external parameters that may be used to setup the model
            mode), #mode set to switch between train, validate and inference mode
            wrt tensorflow.contrib.learn.ModeKeys values
          => the model must return a dictionary of output tensors
-def data_preprocess(features, model_placement)
-def postprocessing_before_export_code(code)
-def postprocessing_before_export_predictions(predictions)
-def getOptimizer(loss, learning_rate, global_step)
-def get_total_loss(inputs, predictions, labels, embedding_code, weights_loss)
-def get_validation_summaries(inputs, predictions, labels, embedding_code)
-def get_eval_metric_ops(inputs, predictions, labels, embedding_code)
-def get_input_pipeline_train_val(batch_size, raw_data_files_folder, shuffle_batches)
-def get_input_pipeline_serving()
-define the Client_IO class that presents at least three methods:
---def __init__(self, debugMode):
''' constructor
    Args:
       debugMode: set True if some debug messages should be displayed
'''

---def getInputData(self, idx):
    ''' method that returns data samples complying with the placeholder
    defined in function get_input_pipeline_serving
    Args:
       idx: the input data index
    Returns:
       the data sample with shape and type complying with the server input
    '''
---def decodeResponse(self, result):
    ''' receive the server response and decode as requested
        have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
        have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
        Args:
        result: a PredictResponse object that contains the request result
    '''
---def finalize(self):
    ''' a function called when the prediction loop ends '''

Here are some examples of such functions:
def data_preprocess(features, model_placement):
    ''' define here the chosen data preprocessing that will be applied
    all the time, for training, validation and serving
    Manually specify here on which device this preprocessing should be done.
    For convenience, the placement of the model that follows this step is also provided
    so that you may want to place it on the same device.
    Args:
        features: the input data that is being processed
        model_placement: the device where the following model will be placed
    Returns:
       the preprocessed data
    '''
    # standardize each column separately
    with tf.device(model_placement):
        mean, var = tf.nn.moments(features, [1], keep_dims=True)
        return tf.div(tf.subtract(features, mean), tf.sqrt(var)+1e-6)

def model_outputs_postprocessing_for_serving(model_outputs_dict):
    ''' define here the post-processings to be applied to each of the model outputs when used withtensorflow serving
        WARNING, in case of multiple outputs, ONE of them must be named as the
        default serving output: tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    Args:
        model_outputs_dict: the original model outputs dictionary
    Returns:
       the postprocessed outputs dictionnary
    '''
    #in this use case, we have two outputs:
    #->  code that is kept as is
    #->  semantic map logits from which we extract the most probable class index for each pixel
    postprocessed_outputs={model_head_embedding_name:model_outputs_dict['code'],
                           model_head_prediction_name:model_outputs_dict['reconstructed_data'],
                           }
    return postprocessed_outputs

def getOptimizer(loss, learning_rate, global_step):
    '''define here the specific optimizer to be used
    '''
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

def get_total_loss(inputs, model_outputs_dict, labels, weights_loss):
    '''a specific loss for data reconstruction when dealing with autoencoders
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
        weights_loss: the model weights loss that may be used for regularization
    '''
    reconstruction_loss=tf.losses.mean_squared_error(
                                model_outputs_dict['reconstructed_data'],
                                inputs,
                                weights=1.0,
                                scope=None,
                                loss_collection=tf.GraphKeys.LOSSES,
                                #reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
                                )

    return reconstruction_loss+weights_weight_decay*weights_loss

def get_validation_summaries(inputs, predictions, labels, embedding_code):
    ''' add here (if required) some summaries to be applied on the validation dataset
    FIXME : to be updated ones validation image summaries become available in future Tensorflow versions
    '''
    labels=tf.squeeze(labels, squeeze_dims=-1)
    semantic_segm_argmax_map=tf.cast(tf.argmax(predictions,3, name='argmax_image'), tf.int32)

    with tf.name_scope('image_summaries'):
        raw_rgb_min= tf.reduce_min(inputs, axis=[1,2,3], keep_dims=True)
        raw_rgb_max= tf.reduce_max(inputs, axis=[1,2,3], keep_dims=True)
        raw_images_rgb_0_1=(inputs-raw_rgb_min)/(raw_rgb_max-raw_rgb_min)
        raw_images_display=tf.saturate_cast(raw_images_rgb_0_1*255.0, dtype=tf.uint8)
        reference_images_crops_display=tf.expand_dims(tf.saturate_cast((labels*255)/nb_classes, dtype=tf.uint8),-1)
        semantic_segm_argmax_map_crops_display=tf.saturate_cast(tf.expand_dims((semantic_segm_argmax_map*255)/nb_classes,-1), dtype=tf.uint8)
        print('*********reference shape='+str(reference_images_crops_display.get_shape().as_list()))
        return [tf.summary.image("input", raw_images_display),
                tf.summary.image("references_center_crop", reference_images_crops_display),
                tf.summary.image("predictions", semantic_segm_argmax_map_crops_display)
               ]

def get_eval_metric_ops(inputs, model_outputs_dict, labels):
    '''Return a dict of the evaluation Ops.
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
    Returns:
        Dict of metric results keyed by name.
    '''
    return {
            'MSE': tf.metrics.mean_squared_error(
                labels=inputs,
                predictions=model_outputs_dict['reconstructed_data'],
                name='mean_squared_error'),
            }

'''Define here the input pipelines :
-1. a common function for train and validation modes
-2. a specific one for the serving model_extra_update_ops
'''
def get_input_pipeline_train_val(batch_size, raw_data_files_folder, shuffle_batches):
    ''' define an input pipeline able to load temporal series from a set of
    CSV files and a batch size specified as inputs
    @param batch_size : the expected size of a batch
    @param raw_data_files_folder : the folder where CSV files are stored
    @param shuffle_batches : a boolean that activates batch shuffling
    '''
    def input_fn():
        #load all csv files to use for training
        raw_data_files=DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_files_folder, file_extension=raw_data_filename_extension)
        print('Input files found='+str(raw_data_files))

        with tf.name_scope("retrieve_data"):
            with tf.device("/cpu:0"):
                data_provider, iterator_initializer_hook=DataProvider_input_pipeline.FileListProcessor_csv_time_series(files=raw_data_files,
                                                                                     record_defaults_values=record_defaults,
                                                                                     nblines_per_block=temporal_series_length,
                                                                                     queue_capacity=batch_size*5,
                                                                                     shuffle_batches=shuffle_batches)
                timestamps, single_period_data_block_raw=data_provider.dequeue_many(batch_size)
                '''
                one label per sample example:
                timestamps_start_stop=tf.string_join([timestamps[:,1],timestamps[:,-1]], separator='->')
                '''
                '''
                two labels per sample example:
                '''
                timestamps_start_stop=tf.stack([timestamps[:,1],timestamps[:,-1]],1)
                #raw_input('timestamps_start_stop='+str(timestamps_start_stop))
        return single_period_data_block_raw, timestamps_start_stop
    return input_fn, None
'''
################################################################################
## Serving (production) section, define here :
-get_input_pipeline_serving():  the input placeholder of the server that will receive the data
-getDataSample_serving(idx): the data samples that you need to send to the server
-received_prediction_serving(result): how to decode the request answer coming from the server

'''
def get_input_pipeline_serving():
    '''Build the serving inputs, expecting messages made of :
    -> a batch of size 1.
    -> a data buffer of type float32 of the same shape as each of the elements used along training (no preliminary normalisation is expected)
    '''
    serialized_tf_example = tf.placeholder(
        dtype=tf.float32,
        shape=[1, temporal_series_length, len(record_defaults)-1],
        name='serialized_input_data')

    return tf.estimator.export.ServingInputReceiver(
        serialized_tf_example, {input_data_name: serialized_tf_example})

class Client_IO:
    ''' A specific class dedicated to clients that need to interract with
    a Tensorflow server that runs the above model
    --> must have the following methods:
    def __init__(self, debugMode): constructor that receives a debug flag
    def getInputData(self, idx): that generates data to send to the server
    def decodeResponse(self, result): that receives the response
    '''
    def __init__(self, debugMode):
        ''' constructor
            Args:
               debugMode: set True if some debug messages should be displayed
        '''
        self.debugMode=debugMode
        if self.debugMode is True:
            print('RPC Client ready to interract with the server')

    def getInputData(self, idx):
        ''' method that returns data samples complying with the placeholder
        defined in function get_input_pipeline_serving
        Args:
           idx: the input data index
        Returns:
           the data sample with shape and type complying with the server input
        '''
        #here, only random numbers
        sample=np.random.random([1,240,12]).astype(np.float32)
        if self.debugMode is True:
            print('Generating input features (random values) of shape '+str(sample.shape))
        return sample


    def decodeResponse(self, result):
        ''' receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        '''
        response = np.array(result.outputs[served_head].float_val)
        print('Answer shape='+str(response.shape))

    def finalize(self):
        ''' a function called when the prediction loop ends '''
        print('Prediction process ended successfuly')

This demo relies on Tensorflow 1.4.1 and makes use of the Experiment and Estimator
available in the contrib module that is expected to evolve along tensorflow versions.
Look at https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/tensorflowcore/trainer/model.py
Look at some general guidelines on Tenforflow here https://github.com/vahidk/EffectiveTensorflow
Look at the related webpages : http://python.usyiyi.cn/documents/effective-tf/index.html
Tensorflow trained graphs can be optimized for inference, some tutorials such as the following may help: https://dato.ml/tensorflow-mobile-graph-optimization/
