'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
'''
# python 2&3 compatibility management
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import DataProvider_input_pipeline
import tensorflow as tf
import numpy as np
import os

from helpers import loss as helpers_loss
from helpers import model as helpers_model

#-> set here your own working folder
workingFolder='experiments/semantic_segmentation'

#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=10#timeout limit when a client requests a served model
#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='Cityscapes'

''' define here some hyperparameters to adjust the experiment
===> Note that this dictionnary will complete the session name
'''
hparams={'nbClasses':34,#set the number of classes in the considered dataset
         'smoothedParams':True,
         'nbEpoch':50}

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
#set here XLA optimisation flags, either tf.OptimizerOptions.OFF#ON_1#OFF
XLA_FLAG=tf.OptimizerOptions.ON_1#OFF

#-> define here the used model under variable 'model'
model_file='examples/segmentation/model_densenet_2D.py'
display_model_layers_info=False #a flag to enable the display of additionnal console information on the model properties (for debug purpose)

field_of_view=109
test_patch_overlapping_ratio=0.75 #-> patch overlapping when evaluating/predicting

#-> define here a string name used for the train, eval and served models
input_data_name='input'
model_head_embedding_name='code'
model_head_prediction_name=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
served_head=model_head_prediction_name #define here the output that will be provided by tensorflow-server

#-> set the number of summaries store per training epoch (more=more precise BUT higer cost)
nb_summary_per_train_epoch=4

#define image patches extraction parameters
patchSize=224
server_patch_size=256
server_crops_per_batch=4

#random seed used to init weights, etc. Use an integer value to make experiments reproducible
random_seed=42

# learning rate decaying parameters
nbEpoch=hparams['nbEpoch']
weights_weight_decay=0.0001
initial_learning_rate=0.0001
num_epochs_per_decay=10 #number of epoch keepng the same learning rate
learning_rate_decay_factor=0.1 #factor applied to current learning rate when NUM_EPOCHS_PER_DECAY is reached
predict_using_smoothed_parameters=hparams['smoothedParams']#set True to use trained parameters values smoothed along the training steps (better results expected BUT STILL DOES NOT WORK WELL IN THIS CODE VERSION)
#set here paths to your data used for train, val, testraw_data_dir_train = "/home/alben/workspace/Datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/"
#-> a first set of data
raw_data_dir_train_ = "/home/alben/workspace/Datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/"
reference_data_dir_train_ = "/home/alben/workspace/Datasets/CityScapes/gtFine_trainvaltest/gtFine/train/"
raw_data_dir_train=(raw_data_dir_train_, reference_data_dir_train_)
raw_data_dir_val_ = "/home/alben/workspace/Datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/val/"
reference_data_dir_val_ = "/home/alben/workspace/Datasets/CityScapes/gtFine_trainvaltest/gtFine/val/"
raw_data_filename_extension='*.png'
ref_data_filename_extension='*labelIds.png'
#load all image files to use for training or testing
nb_train_images=len(DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_dir_train_, file_extension=raw_data_filename_extension, raiseOnEmpty=False))
nb_val_images=len(DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_dir_val_, file_extension=raw_data_filename_extension, raiseOnEmpty=False))
reference_labels=['semantic_labels']

raw_data_dir_val=(raw_data_dir_val_, reference_data_dir_val_)
number_of_crops_per_image=100
nb_train_samples=nb_train_images*number_of_crops_per_image# number of images * number of crops per image
nb_test_samples=7000#nb_val_images*number_of_crops_per_image
eval_not_rerun_until_sec=3600*12 #wait for 12h between 2 evaluations (long runs)
batch_size=4

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
    # do nothing, train and val input paipeline standardize data on their own and
    # serving will do its own too
    return features

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
    postprocessed_outputs={#model_head_embedding_name:model_outputs_dict['code'],
                           model_head_prediction_name:tf.saturate_cast(tf.argmax(model_outputs_dict['logits_semantic_map'],3, name='argmax_image'), tf.uint8),
                           }
    return postprocessed_outputs

def getOptimizer(loss, learning_rate, global_step):
    '''define here the specific optimizer to be used
    '''
    #get gradient summary information and the gradient norm
    tvars, raw_grads, gradient_norm=helpers_model.track_gradients(loss)

    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

def get_total_loss(inputs, model_outputs_dict, labels, weights_loss):
    '''a specific loss for data reconstruction when dealing with autoencoders
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
        weights_loss: the model weights loss that may be used for regularization
    '''
    labels=tf.squeeze(labels, axis=[-1])

    #-> restrict to the center part of the images
    logits_semantic_crops=tf.slice(model_outputs_dict['logits_semantic_map'], begin=[0,field_of_view//2, field_of_view//2, 0], size =[-1,patchSize-field_of_view, patchSize-field_of_view, -1])
    cross_entropy_segmentation_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_semantic_crops, labels=labels))

    return cross_entropy_segmentation_loss+weights_weight_decay*weights_loss

def get_validation_summaries(inputs, model_outputs_dict, labels):
    ''' add here (if required) some summaries to be applied on the validation dataset
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
    Returns:
        a list of summaries
    '''
    labels=tf.squeeze(labels, squeeze_dims=-1)
    semantic_segm_argmax_map=tf.cast(tf.argmax(model_outputs_dict['logits_semantic_map'],3, name='argmax_image'), tf.int32)

    with tf.name_scope('image_summaries'):
        raw_rgb_min= tf.reduce_min(inputs, axis=[1,2,3], keepdims=True)
        raw_rgb_max= tf.reduce_max(inputs, axis=[1,2,3], keepdims=True)
        raw_images_rgb_0_1=(inputs-raw_rgb_min)/(raw_rgb_max-raw_rgb_min)
        raw_images_display=tf.saturate_cast(raw_images_rgb_0_1*255.0, dtype=tf.uint8)
        reference_images_crops_regions_display=tf.expand_dims(tf.saturate_cast((labels*255)/(hparams['nbClasses']-1), dtype=tf.uint8),-1)
        semantic_segm_argmax_map_crops_display=tf.saturate_cast(tf.expand_dims((semantic_segm_argmax_map*255)/(hparams['nbClasses']-1),-1), dtype=tf.uint8)
        print('*********reference shape='+str(reference_images_crops_regions_display.get_shape().as_list()))
        return ([tf.summary.image("input", raw_images_display),
                tf.summary.image("references_center_crop_regions", reference_images_crops_regions_display),
                tf.summary.image("semantic_segmentation", semantic_segm_argmax_map_crops_display),
               ], nb_test_samples/4)

def get_eval_metric_ops(inputs, model_outputs_dict, labels):
    """Return a dict of the evaluation Ops.
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
    Returns:
        Dict of metric results keyed by name.
    """
    labels=tf.squeeze(labels, squeeze_dims=-1)

    semantic_segm_argmax_map=tf.cast(tf.argmax(model_outputs_dict['logits_semantic_map'],3, name='argmax_image'), tf.int32)
    print('Evaluating on the center part of the window (skeeping border effects, TODO, adjust size wrt receptive field shape)')
    print('-> semantic map shape={seg_shp}, ref_shape={ref_shp}'.format(seg_shp=semantic_segm_argmax_map.get_shape().as_list(),
                                                                    ref_shp=labels.get_shape().as_list()))

    return {
            'Accuracy': tf.metrics.accuracy(
                        labels=labels,
                        predictions=semantic_segm_argmax_map,
                        weights=None,
                        metrics_collections=None,
                        updates_collections=None,
                        name='Accuracy_metric'
                        ),
            'IoU' : tf.metrics.mean_iou(
                                labels=labels,
                                predictions=semantic_segm_argmax_map,
                                num_classes=hparams['nbClasses'],
                                weights=None,
                                metrics_collections=None,
                                updates_collections=None,
                                name='IoU_metric')
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

    class IteratorInitializerHook(tf.train.SessionRunHook):
        """Hook to initialise data iterator after Session is created."""

        def __init__(self):
            super(IteratorInitializerHook, self).__init__()
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            """Initialise the iterator after the session has been created."""
            self.iterator_initializer_func(session)
    iterator_initializer_hook = IteratorInitializerHook()

    if nb_train_images==0:
        raise ValueError('No training image found, abording!')
    if nb_val_images==0:
        raise ValueError('No training image found, abording!')
    # get model field of view computed at the training step or compute it with the test_patch_overlapping_ratio
    def get_fov(isTraining):
        fov=0
        if isTraining is False:
            if test_patch_overlapping_ratio < 1.0 and test_patch_overlapping_ratio > 0.0 :
                fov = int(np.ceil(test_patch_overlapping_ratio*patchSize) // 2 * 2 + 1)
            elif test_patch_overlapping_ratio == 0 :
                fov = 0
            else :
                fov = field_of_view
        return fov

    import cv2
    isTraining=shuffle_batches
    def apply_pixel_transforms(isTraining):
        if isTraining:
            return 0.5
        else:
            return None
    def input_fn():
        #load all image files to use for training or testing
        raw_data_files=DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_files_folder[0], file_extension=raw_data_filename_extension)
        dataset_references_train=DataProvider_input_pipeline.extractFilenames(raw_data_files_folder[1], ref_data_filename_extension)
        nb_raw_images=len(raw_data_files)
        nb_ref_images=len(dataset_references_train)
        print('Input files found (raw, ref)='+str((nb_raw_images, nb_ref_images)))
        if nb_raw_images!=nb_ref_images:
            raise ValueError('Raw images and reference image numbers differ, check datasets')

        #init the input pipeline
        dataset_nbEpoch=nbEpoch#for the training dataset
        if isTraining is False:
          dataset_nbEpoch=1
        data_provider=DataProvider_input_pipeline.FileListProcessor_Semantic_Segmentation(raw_data_files, dataset_references_train,
                nbEpoch=dataset_nbEpoch,
                shuffle_samples=shuffle_batches,
                patch_ratio_vs_input=patchSize,
                max_patches_per_image=number_of_crops_per_image,
                image_area_coverage_factor=int(isTraining)+1.0,#factor 2 on training, 1 on testing
                num_preprocess_threads=4,#4 threads on training, 1 on testing
                apply_random_flip_left_right=isTraining,
                apply_random_flip_up_down=False,
                apply_random_brightness=apply_pixel_transforms(isTraining),
                apply_random_saturation=apply_pixel_transforms(isTraining),
                apply_whitening=True,
                batch_size=batch_size,
                use_alternative_imread='opencv',
                balance_classes_distribution=isTraining,
                classes_entropy_threshold=0.6,
                opencv_read_flags=cv2.IMREAD_UNCHANGED,
                field_of_view=get_fov(isTraining),
                manage_nan_values='avoid')

        #retreive a batch of samples
        with tf.name_scope("retrieve_batch"):
            # batch sample retrieval
            data_batch=data_provider.dataset_iterator.get_next()
            # extract raw data,  reference data will be extracted at the optimizer level
            raw_images=tf.slice( data_batch,
                                        begin=[0,0,0,0],
                                        size=[-1,-1,-1,data_provider.single_image_raw_depth])
            with tf.name_scope('prepare_reference_data'):
                #-> get reference data restricted to the center part of the images
                reference_crops=tf.cast(
                                        tf.slice( data_batch,
                                            begin=[0,field_of_view//2, field_of_view//2, data_provider.single_image_raw_depth],
                                            size=[-1,patchSize-field_of_view, patchSize-field_of_view,data_provider.single_image_reference_depth])
                                    ,dtype=tf.int32)
        #finally manage the dataset iterator
        iterator_init = data_provider.getIteratorInitializer()
        # Set runhook to initialize the iterator
        iterator_initializer_hook.iterator_initializer_func = \
            lambda sess: sess.run(iterator_init)
        return raw_images, reference_crops
    return input_fn, iterator_initializer_hook
'''
################################################################################
## Serving (production) section, define here :
-get_input_pipeline_serving():  the input placeholder of the server that will receive the data
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
'''
serving_img_shape=[server_crops_per_batch, server_patch_size,server_patch_size,3]
def get_input_pipeline_serving():
    '''Build the serving inputs, expecting messages made of :
    -> a batch of size 1 of a single image in the uint8 format (no preliminary normalisation is expected).
    ---> the input is then converted into a float32 4D batch
    '''
    serialized_tf_example = tf.placeholder(
        dtype=tf.uint8,
        shape=serving_img_shape,
        name='serialized_input_data')
    serialized_tf_example_4d=tf.map_fn(tf.image.per_image_standardization, tf.cast(serialized_tf_example, dtype=tf.float32))
    print('Served input='+str(serialized_tf_example_4d))
    return tf.estimator.export.ServingInputReceiver(
        serialized_tf_example_4d, {input_data_name: serialized_tf_example})


import cv2
class Client_IO:
    ''' A specific class dedicated to clients that need to interract with
    a Tensorflow server that runs the above model
    --> must have the following methods:
    def __init__(self, clientInitSpecs, debugMode): constructor that receives a debug flag
    def getInputData(self, idx): that generates data to send to the server
    def decodeResponse(self, result): that receives the response
    def finalize(self): the method call at the end of the process
    '''
    def __init__(self, clientInitSpecs={}, debugMode=False):
        ''' constructor
            Args:
               clientInitSpecs: a dictionnary to setup the client is necessary
               debugMode: set True if some debug messages should be displayed
        '''
        self.onlineDisplay=False #set True to activate online imshow (maybe to show progress)
        self.outputFilename_prefix='segmentation.out'
        self.debugMode=debugMode
        self.input_stream=None
        self.input_sample=None
        self.crops_positions_in=[]
        self.output_path=''
        if 'output_path' in clientInitSpecs.keys():
            print('Output predictions will be written here: '+str(clientInitSpecs['output_path']))
            self.output_path=clientInitSpecs['output_path']
        if 'sample_file' in clientInitSpecs.keys():
          print('sample_file provided: '+str(clientInitSpecs['sample_file']))
          self.inframe=cv2.imread(clientInitSpecs['sample_file'])
          self.resize_input_to_model_expectation=False
          self.outputFilename_prefix=os.path.basename(clientInitSpecs['sample_file'])+'.'+self.outputFilename_prefix
        else:
          self.input_stream=cv2.VideoCapture(0)
          valid, self.inframe=self.input_stream.read()

        #prepare crops to process the entire input image iteratively

        if self.onlineDisplay:
          cv2.imshow('input', self.inframe)
        #prepare an output image buffer with the same size as the original input (before padding)
        self.outframe=hparams['nbClasses']*np.ones((self.inframe.shape[0], self.inframe.shape[1]), dtype=np.uint8)
        #add padding to handle borders
        #self.inframe=np.pad(array=self.inframe, pad_width=((field_of_view//2,field_of_view//2), (field_of_view//2,field_of_view//2), (0,0)), mode='constant')
        print('(Field of view size, Padded input image shape)='+str((field_of_view, self.inframe.shape)))
        #define crops parsing indicators
        self.crop_index=0
        self.patch_effective_width=server_patch_size-field_of_view

        #get the number of patch rows and cols on the original input image size (same as output frame)
        self.nb_crops_lines=np.maximum(1,(self.outframe.shape[0]-field_of_view)//self.patch_effective_width)
        self.nb_crops_colums=np.maximum(1,(self.outframe.shape[1]-field_of_view)//self.patch_effective_width)
        self.crops_positions_in=[]
        self.crops_positions_code=[]
        for l in six.moves.range(self.nb_crops_lines):
          for c in six.moves.range(self.nb_crops_colums):
            self.crops_positions_in.append((l*self.patch_effective_width,c*self.patch_effective_width))
        #adding bottom and right borders
        for l in six.moves.range(self.nb_crops_lines):
          self.crops_positions_in.append((l*self.patch_effective_width,self.outframe.shape[1]-server_patch_size))
        for c in six.moves.range(self.nb_crops_colums):
          self.crops_positions_in.append((self.outframe.shape[0]-server_patch_size,c*self.patch_effective_width))
        self.crops_positions_in.append((self.outframe.shape[0]-server_patch_size,self.outframe.shape[1]-server_patch_size))
        print('Number of crops required to process the input image = '+str(len(self.crops_positions_in)))

        if self.debugMode is True:
            print('RPC Client ready to interract with the server')
            print('Image reader ready, original frame size='+str(self.inframe.shape))

        # reporting here the Cityscapes labels lookup table to use on the
        # client side for semantic map color visualisation.
        #-> From https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        # a label and all meta information
        from collections import namedtuple
        Label = namedtuple( 'Label' , [

            'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                            # We use them to uniquely name a class

            'id'          , # An integer ID that is associated with this label.
                            # The IDs are used to represent the label in ground truth images
                            # An ID of -1 means that this label does not have an ID and thus
                            # is ignored when creating ground truth images (e.g. license plate).
                            # Do not modify these IDs, since exactly these IDs are expected by the
                            # evaluation server.

            'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                            # ground truth images with train IDs, using the tools provided in the
                            # 'preparation' folder. However, make sure to validate or submit results
                            # to our evaluation server using the regular IDs above!
                            # For trainIds, multiple labels might have the same ID. Then, these labels
                            # are mapped to the same class in the ground truth images. For the inverse
                            # mapping, we use the label that is defined first in the list below.
                            # For example, mapping all void-type classes to the same ID in training,
                            # might make sense for some approaches.
                            # Max value is 255!

            'category'    , # The name of the category that this label belongs to

            'categoryId'  , # The ID of this category. Used to create ground truth images
                            # on category level.

            'hasInstances', # Whether this label distinguishes between single instances or not

            'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                            # during evaluations or not

            'color'       , # The color of this label
        ] )

        self.labels = [
            #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
            Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
            Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
            Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
            Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
            Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
            Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
            Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
            Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
            Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
            Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
            Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
            Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
            Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
            Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
            Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
            Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
            Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
            Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
            Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
            Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
            Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
            Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
            Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
            Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
            Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
            Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
            Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
            Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
            Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
            Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
        ]
        #--------------------------------------------------------------------------------
        # Create dictionaries for a fast lookup
        #--------------------------------------------------------------------------------
        # Please refer to the main method below for example usages!

        # name to label object
        self.label_names      = { label.name    : label for label in self.labels           }
        print('label_names=',self.label_names.keys())
        # id to color label object
        self.cityscapes_labels_colors=np.zeros((256,1,3), dtype=np.uint8)
        self.cityscapes_labels_colors[0:35] = np.reshape(np.array([ label[7] for label in self.labels ], dtype=np.uint8), (35,1,3))

        if self.debugMode is True:
            print('cityscapes_labels_colors'+str(self.cityscapes_labels_colors))
            print('cityscapes_labels_colors.shape'+str(self.cityscapes_labels_colors.shape))
            print('cityscapes_labels_colors[5]='+str(self.cityscapes_labels_colors[5]))


    def getInputData(self, idx):
        ''' method that returns data samples complying with the placeholder
        defined in function get_input_pipeline_serving
        Args:
           idx: the input data index
        Returns:
           the data sample with shape and type complying with the server input
        '''

        ''' process image crop by crop '''
        if self.crop_index>=len(self.crops_positions_in):
          #TODO, exit the process
          print('Input image has been fully parsed, program end...')
          raise StopIteration

        crops=[]
        #self.crop_index is not updated here, decodeResponse will do this FIXME, any async issue ?
        crop_index=self.crop_index
        for i in six.moves.range(server_crops_per_batch):
          #end of the list border effect management (add a zeroes patch
          if crop_index+i>=len(self.crops_positions_in):
            current_crop=np.zeros(serving_img_shape[1:], dtype=np.uint8)
          else:
            crop_coord=self.crops_positions_in[crop_index+i]
            #print('Preparing request batch with crop (index, coord)='+str((crop_index+i, crop_coord)))
            current_crop=self.inframe[crop_coord[0]:crop_coord[0]+server_patch_size,crop_coord[1]:crop_coord[1]+server_patch_size,:]
          # put crop into the batch
          crops.append(current_crop)
            #print('Added crop of shape : '+str(current_crop.shape))
        #self.crops_positions
        self.frame_patch=np.array(crops)
        if self.debugMode is True:
          print('Batch request shape={shp}, crop index={cropIdx}'.format(shp=self.frame_patch.shape, cropIdx=crop_index))
          for idx in six.moves.range(server_crops_per_batch):
            cv2.imshow('input crop'+str(idx), self.frame_patch[idx,:,:,:])
          cv2.waitKey(5)

        if self.debugMode is True:
            print('Input frame'+str(self.frame_patch.shape))
        return self.frame_patch

    def decodeResponse(self, result):
        ''' receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        '''
        response = np.reshape(np.array(result.outputs[served_head].int_val),serving_img_shape[:-1]).astype(np.uint8)

        if self.debugMode is True:
            print('Answer shape='+str(response.shape))
            print('self.frame_patch.shape'+str(self.frame_patch.shape))

        #self.add_segmentation_colors(response, self.frame_patch)
        if self.onlineDisplay:
          cv2.imshow('Semantic segmentation overlay', self.frame_patch)

        #self.crop_index is not updated here, decodeResponse will do this FIXME, any async issue ?
        fov_half=(field_of_view-1)//2
        crop_index=self.crop_index
        for i in range(server_crops_per_batch):
          #end of the list management : avoid last batch samples that do not fit into the image
          if crop_index+i>=len(self.crops_positions_in):
            break
          crop_coord=self.crops_positions_in[crop_index+i]
          row_start_idx=crop_coord[0]+fov_half
          row_stop_idx=row_start_idx+self.patch_effective_width
          col_start_idx=crop_coord[1]+fov_half
          col_stop_idx=col_start_idx+self.patch_effective_width
          if self.debugMode is True:
            print('*** received patch:'+str((i, crop_coord)))
            print('central patch bbox (y,y+h, x, x+w)='+str((row_start_idx,row_stop_idx,col_start_idx,col_stop_idx)))
            print('in_row='+str(response.shape))
            print('outframe='+str(self.outframe.shape))
            print('out='+str(self.outframe[row_start_idx:row_stop_idx,col_start_idx:col_stop_idx].shape))
          self.outframe[row_start_idx:row_stop_idx,col_start_idx:col_stop_idx]=response[i,fov_half:fov_half+self.patch_effective_width,fov_half:fov_half+self.patch_effective_width]
        #finaly update the batch crop index
        self.crop_index+=server_crops_per_batch

        if self.onlineDisplay:
          cv2.imshow("reconstruction", (self.outframe.astype(np.int16)*255//hparams['nbClasses']).astype(np.uint8))
          cv2.waitKey(4)
        else:
          print('prediction step {step}/{total}'.format(step=crop_index, total=len(self.crops_positions_in)))


    def add_segmentation_colors(self, segmentation_map, outputImage):
        semantic_map_3c=cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2BGR);
        semantic_map_color=cv2.LUT(semantic_map_3c, self.cityscapes_labels_colors)
        #applying the semantic segmentation map as an overlay on the input
        alpha=0.5
        cv2.addWeighted(semantic_map_color, alpha, outputImage, 1 - alpha, 0, outputImage)


    def finalize(self):
        ''' a function called when the prediction loop ends '''
        print('Prediction process ended successfuly, press a key to close')
        file_path_prefix=os.path.join(self.output_path, self.outputFilename_prefix)
        print('output images prefix='+str(file_path_prefix))
        self.add_segmentation_colors(self.outframe, self.inframe)
        cv2.imwrite(file_path_prefix+'_pred.bmp', self.outframe)
        cv2.imwrite(file_path_prefix+'_overlay.bmp', self.inframe)
