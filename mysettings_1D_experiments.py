'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
'''
import DataProvider_input_pipeline
import tensorflow as tf
import numpy as np

#-> set here your own working folder
workingFolder='experiments/1Dsignals_clustering'

#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=1#timeout limit when a client requests a served model


#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='my_test'

#-> allow X window displays (for image and graph display purpose)
allow_display=True

#-> activate session profiling to observe ressource use and timings
do_trace_computation=True

''''set the list of GPUs involved in the process. HOWTO:
->if using CPU only mode, let an empty list
->if using a single GPU, only the first ID of the list will be considered
->if using multiple GPUs, each GPU ID will be considered
=> general recommendation: always try to focus on unused GPUs to avoid conflicts
with other processing jobs, yours and the ones of your colleagues.
Then, connect to the processing node and type in command line 'nvidia-smi'
to check which gpu is free (very few used memory and GPU )
'''
used_gpu_IDs=[0]

#-> define here the used model under variable 'model'
#model_file='model_densenet.py'
model_file='model_densenet_1D.py'
field_of_view=20

display_model_layers_info=False #a flag to enable the display of additionnal console information on the model properties (for debug purpose)
#-> define here a string name used for the train, eval and served models
input_data_name='input'
model_head_embedding_name='code'
model_head_prediction_name=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY#'prediction'
#->define here the output that will be provided by tensorflow-server
served_head=model_head_embedding_name

#-> define the training strategy depending on the computing architecture
#---> "continuous_train_and_eval"-> single machine
#---> "train_and_evaluate" -> multiple machines/distributed training/evaluation
train_val_schedule_strategy="continuous_train_and_eval"

#-> set the number of summaries store per training epoch (more=more precise BUT higer cost)
nb_summary_per_train_epoch=4

#define image patches extraction parameters
patchSize=224

# learning rate decaying parameters
nbEpoch=300
weights_weight_decay=0.0001
initial_learning_rate=0.001
num_epochs_per_decay=150 #number of epoch keepng the same learning rate
learning_rate_decay_factor=0.1 #factor applied to current learning rate when NUM_EPOCHS_PER_DECAY is reached
predict_using_smoothed_parameters=False#set True to use trained parameters values smoothed (EMA) along the training steps (better results expected BUT STILL DOES NOT WORK WELL IN THIS CODE VERSION)

#set here paths to your data used for train, val
#-> a first set of data
raw_data_dir_train = "datasamples/fakedata/train"
raw_data_dir_val = "datasamples/fakedata/val"
temporal_series_length=240
nb_train_samples=148663/temporal_series_length #manually adjust here the number of temporal items out of the temporal block size
nb_test_samples=58352/temporal_series_length
record_defaults=[['timestamp'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ]
reference_labels=['startDate', 'stopDate'] #to be used if many labels are generated by the get_input_pipeline_train_val function
batch_size=20
nb_classes=23
raw_data_filename_extension='/*.csv'
ref_data_filename_extension='/*.csv'
csv_field_delim=','

####################################################
## Define here use case specific metrics, loss, etc.
#with tf.name_scope("loss"):
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
    TODO, look at the doc here : https://www.tensorflow.org/programmers_guide/datasets
    @param batch_size : the expected size of a batch
    @param raw_data_files_folder : the folder where CSV files are stored
    @param shuffle_batches : a boolean that activates batch shuffling
    '''
    def input_fn():
        #load all csv files to use for training
        raw_data_files=DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_files_folder, file_extension=raw_data_filename_extension)
        print('Input files found='+str(raw_data_files))

        with tf.name_scope("retrieve_data"):
            data_provider, iterator_initializer_hook=DataProvider_input_pipeline.FileListProcessor_csv_time_series(files=raw_data_files,
                                                                                 csv_field_delim=csv_field_delim,
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
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
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
        self.sample=np.random.random([1,240,12]).astype(np.float32)
        if self.debugMode is True:
            print('Generating input features (random values) of shape '+str(self.sample.shape))
        return self.sample

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
