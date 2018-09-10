'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs

Some notes on the data fom Kevin Jacq(Edytem):
canal des lamines : 0=pas de labels et non annotie, 1=lamine hiver, 2=lamineete

canal crues : 0=pas de labels et non annote, 1=crues (la j'ai choisi la zone crue avec un rectangle, mais ce n'est pas le cas, donc le voisinage peut etre des pixels de crues, ensuite il peut avoir des crues plus fines que je ne connais pas)
canal lamines et crues : 0=pas de labels et non annote, 1=lamine hiver, 2=lamine ete, 3=crues (j'ai quand meme ce le chiffre)

Autre approche : 20 points d'analyse chimique par carotte (taux de matiere organique).
'''
import DataProvider_input_pipeline
import tensorflow as tf
import numpy as np

#-> set here your own working folder
workingFolder='experiments/hyperspectral_images'

#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=20#timeout limit when a client requests a served model
#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='Carottes_edytem_DenseNet'

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
XLA_FLAG=tf.OptimizerOptions.OFF#ON_1#OFF

#-> define here the used model under variable 'model'
#model_file='model_densenet.py'
model_file='model_densenet_3D.py'
display_model_layers_info=False #a flag to enable the display of additionnal console information on the model properties (for debug purpose)

field_of_view=9
test_patch_overlapping_ratio=0.75 #-> patch overlapping when evaluating/predicting

#-> define here a string name used for the train, eval and served models
input_data_name='input'
model_head_embedding_name='code'
model_head_prediction_name=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
served_head=model_head_prediction_name #define here the output that will be provided by tensorflow-server

#-> set the number of summaries store per training epoch (more=more precise BUT higer cost)
nb_summary_per_train_epoch=4

#define image patches extraction parameters
patchSize=32
server_patch_size=32#let's try with a different patch size when serving
server_crops_per_batch=4#define here how many crops are sent to the server at a single time

#random seed used to init weights, etc. Use an integer value to make experiments reproducible
random_seed=42

# learning rate decaying parameters
nbEpoch=50
weights_weight_decay=0.0001
initial_learning_rate=0.001
num_epochs_per_decay=20 #number of epoch keepng the same learning rate
learning_rate_decay_factor=0.1 #factor applied to current learning rate when NUM_EPOCHS_PER_DECAY is reached
predict_using_smoothed_parameters=False#set True to use trained parameters values smoothed along the training steps (better results expected BUT STILL DOES NOT WORK WELL IN THIS CODE VERSION)
#set here paths to your data used for train, val, testraw_data_dir_train = "/home/alben/workspace/Datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/"
#-> a first set of data
raw_data_dir_train = "/home/alben/workspace/Datasets/hyperspectral/carottes/train/SWIR/"#"/uds_data/listic/datasets/hyperspectral/carottes/train/SWIR/"
raw_data_dir_val = "/home/alben/workspace/Datasets/hyperspectral/carottes/train/SWIR/"#"/uds_data/listic/datasets/hyperspectral/carottes/val/SWIR/"
raw_data_filename_extension='*.tif'
ref_data_filename_extension='*.tif'
#load all image files to use for training or testing
nb_train_images=len(DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_dir_train, file_extension=raw_data_filename_extension))
nb_val_images=len(DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_dir_val, file_extension=raw_data_filename_extension))
reference_labels=['inconnu_lamine_crue']
number_of_crops_per_image=200
nb_train_samples=nb_train_images*number_of_crops_per_image#nb_train_images*number_of_crops_per_image# number of images * number of crops per image
nb_test_samples=1*nb_val_images*number_of_crops_per_image
batch_size=4
nb_classes=10
input_nb_spectral_bands=128#specify here the number of spectral band (central bands) that should be considered for processing
first_selected_band_id=7#the index of the first band to process (follows the input_nb_spectral_bands)
#REMINDER : skip the first and last 10 bands
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
                           model_head_prediction_name:model_outputs_dict['code']#tf.saturate_cast(model_outputs_dict['reconstructed_data'], tf.uint16),
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
    print('inputs='+str(inputs))
    print('recons='+str(model_outputs_dict['reconstructed_data']))
    reconstruction_loss=tf.losses.mean_squared_error(
                                    model_outputs_dict['reconstructed_data'],
                                    inputs,
                                    weights=1.0,
                                    scope=None,
                                    loss_collection=tf.GraphKeys.LOSSES,
                                    #reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
                                    )

    tf.summary.scalar('weights_loss', weights_loss)
    tf.summary.scalar('mse_loss', reconstruction_loss)
    return reconstruction_loss+weights_weight_decay*weights_loss

def get_eval_metric_ops(inputs, model_outputs_dict, labels):
    """Return a dict of the evaluation Ops.
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
    Returns:
        Dict of metric results keyed by name.
    """

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
    @param raw_data_files_folder : the folder where files are stored
    @param shuffle_batches : a boolean that activates batch shuffling
    '''

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
        print('raw_data_files_folder='+str(raw_data_files_folder))
        raw_data_files=DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_files_folder, file_extension=raw_data_filename_extension)

        #init the input pipeline
        data_provider=DataProvider_input_pipeline.FileListProcessor_Semantic_Segmentation(raw_data_files, -1,
                shuffle_samples=True,#shuffle_batches,
                patch_ratio_vs_input=patchSize,
                max_patches_per_image=number_of_crops_per_image,
                image_area_coverage_factor=int(isTraining)+1.0,#factor 2 on training, 1 on testing
                num_preprocess_threads=1,#4 threads on training, 1 on testing
                apply_random_flip_left_right=False,#isTraining,
                apply_random_flip_up_down=False,
                apply_random_brightness=None,#apply_pixel_transforms(isTraining),
                apply_random_saturation=None,#apply_pixel_transforms(isTraining),
                apply_whitening=False,
                batch_size_train=batch_size,
                use_alternative_imread='gdal',
                balance_classes_distribution=False,#isTraining,
                classes_entropy_threshold=0.3,
                opencv_read_flags=None,#cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYDEPTH,
                field_of_view=get_fov(isTraining),
                manage_nan_values='avoid')

        #retreive a batch of samples
        last_labels_channels_nb=3
        with tf.name_scope("retrieve_batch"):
            # batch sample retrieval
            data_batch=data_provider.deepnet_data_queue.dequeue_many(batch_size)
            # extract raw data,  reference data will be extracted at the optimizer level
            raw_images=tf.expand_dims(tf.slice( data_batch,
                                        begin=[0,0,0,first_selected_band_id],#int(data_provider.single_image_raw_depth-last_labels_channels_nb-input_nb_spectral_bands)/2],
                                        size=[-1,-1,-1,input_nb_spectral_bands]),-1)
            with tf.name_scope('prepare_reference_data'):
                #-> get reference data restricted to the center part of the images
                reference_crops=tf.expand_dims(tf.cast(
                                        tf.slice( data_batch,
                                            begin=[0,field_of_view/2, field_of_view/2, data_provider.single_image_raw_depth-1],
                                            size=[-1,patchSize-field_of_view, patchSize-field_of_view,1])
                                    ,dtype=tf.int32),
                                    -1)
        return raw_images, reference_crops
    return input_fn, None
'''
################################################################################
## Serving (production) section, define here :
-get_input_pipeline_serving():  the input placeholder of the server that will receive the data
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
'''

serving_img_shape=[server_crops_per_batch, server_patch_size,server_patch_size,input_nb_spectral_bands]
def get_input_pipeline_serving():
    '''Build the serving inputs, expecting messages made of :
    -> a batch of multiple image crops in the uint16 format (no preliminary normalisation is expected).
    ---> the input is then converted into a float32 5D batch and is processed as for trainn/val
    '''
    serialized_tf_example = tf.placeholder(
        dtype=tf.uint16,
        shape=serving_img_shape,
        name='serialized_input_data')

    img_5D=tf.expand_dims(tf.cast(serialized_tf_example, dtype=tf.float32),-1)
    print('Served input='+str(img_5D))
    return tf.estimator.export.ServingInputReceiver(
        img_5D, {input_data_name: serialized_tf_example})


import cv2
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
        #Some test setup here
        test_file="/home/alben/workspace/Datasets/hyperspectral/carottes/train/SWIR/LDBSWIR.tif"

        self.debugMode=debugMode #set True to activate debug prints
        inframe=DataProvider_input_pipeline.imread_from_gdal(test_file, debug_mode=True)
        self.code_scale=8

        #convert to expected server input format (type and keep the expected number of bands only)
        self.inframe=inframe.astype(np.uint16)[:,:,first_selected_band_id:first_selected_band_id+input_nb_spectral_bands]
        self.reference=inframe.astype(np.uint16)[:,:,-3:]#keep the ground truth of the 3 last layers
        #add padding to handle borders
        self.inframe=np.pad(array=self.inframe, pad_width=((field_of_view/2,field_of_view/2), (field_of_view/2,field_of_view/2), (0,0)), mode='constant')
        self.reference=np.pad(array=self.reference, pad_width=((field_of_view/2,field_of_view/2), (field_of_view/2,field_of_view/2), (0,0)), mode='constant')
        #cv2.imread('../../../../datasamples/semantic_segmentation/raw_data/aachen_000000_000019_leftImg8bit.png')
        self.outframe=np.zeros((self.inframe.shape[0], self.inframe.shape[1], self.inframe.shape[2]))
        self.codeframe=np.zeros((self.inframe.shape[0]/self.code_scale, self.inframe.shape[1]/self.code_scale, 512))
        #define crops parsing indicators
        self.crop_index=0
        self.nb_crops_lines=self.outframe.shape[0]/(patchSize-field_of_view)
        self.nb_crops_colums=self.outframe.shape[1]/(patchSize-field_of_view)
        self.crops_positions_in=[]
        self.crops_positions_code=[]
        self.patch_effective_width=server_patch_size-field_of_view
        self.code_effective_width=self.patch_effective_width/self.code_scale
        for l in range(self.nb_crops_lines):
          for c in range(self.nb_crops_colums):
            self.crops_positions_in.append((l*self.patch_effective_width,c*self.patch_effective_width))
            self.crops_positions_code.append((l*self.code_effective_width,c*self.code_effective_width))
        print('Processing taking into account model field of view : patch_effective_width, code_effective_width='+str((self.patch_effective_width, self.code_effective_width)))
        #print('patches in top left='+str(self.crops_positions_in))
        #print('code out   top left='+str(self.crops_positions_code))
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
            Label(  'unlabeled'            ,  0 ,      0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'crues'                ,  1 ,      1 , 'void'            , 1       , False        , True         , (  255,  0,  0) ),
            Label(  'lamine_summer'        ,  2 ,      2 , 'void'            , 2       , False        , True         , (  0,  255,  0) ),
            Label(  'lamine_winter'        ,  3 ,      3 , 'void'            , 2       , False        , True         , (  0,  0,  255) ),
        ]
        #--------------------------------------------------------------------------------
        # Create dictionaries for a fast lookup
        #--------------------------------------------------------------------------------
        # Please refer to the main method below for example usages!

        # name to label object
        name2label      = { label.name    : label for label in self.labels           }
        # id to color label object
        self.carottes=np.zeros((len(self.labels),1,3), dtype=np.uint8)
        self.carottes= np.reshape(np.array([ label[7] for label in self.labels ], dtype=np.uint8), (len(self.labels),1,3))
        if self.debugMode is True:
            print('carottes'+str(self.carottes))
            print('carottes.shape'+str(self.carottes.shape))

    def getInputData(self, idx):
        ''' method that returns data samples complying with the placeholder
        defined in function get_input_pipeline_serving
        Args:
           idx: the input data index
        Returns:
           the data sample with shape and type complying with the server input
        '''
        if self.crop_index>=len(self.crops_positions_in):
          #TODO, exit the process
          print('Input image has been fully parsed, program end...')
          raise StopIteration

        crops=[]
        #self.crop_index is not updated here, decodeResponse will do this FIXME, any async issue ?
        crop_index=self.crop_index
        for i in range(server_crops_per_batch):
          crop_coord=self.crops_positions_in[i]
          crops.append(self.inframe[crop_coord[0]:crop_coord[0]+server_patch_size,crop_coord[1]:crop_coord[1]+server_patch_size,:])
          crop_index+=1

        #self.crops_positions
        self.frame_patch=np.array(crops)
        print('Crops to be sent to server shape='+str(self.frame_patch.shape))
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
        response = np.reshape(np.array(result.outputs[model_head_prediction_name].float_val),
                              [server_crops_per_batch, 4, 4, 16*32]).astype(np.float32)
        print('Received answer shape '+str(response.shape))

        #self.crop_index is not updated here, decodeResponse will do this FIXME, any async issue ?
        crop_index=self.crop_index
        for i in range(server_crops_per_batch):
          crop_coord=self.crops_positions_code[self.crop_index+i]
          print(crop_coord)
          self.codeframe[crop_coord[0]:crop_coord[0]+2,crop_coord[1]:crop_coord[1]+2,:]=response[i,1:3,1:3,:]
          self.crops_positions_code
          crop_index+=1
        #finaly update the batch crop index
        self.crop_index+=server_crops_per_batch

        #TODO, do some clustering to draw a nice image...
        cv2.imshow("code", self.codeframe[:,:,:3].astype(np.uint8))
        cv2.waitKey(4)

    def finalize(self):
        ''' a function called when the prediction loop ends '''
        print('Prediction process ended successfuly')
        np.savetxt('/home/alben/code.csv', self.codeframe)
        if self.debugMode is True:
            print('Answer shape='+str(response.shape))

        cv2.imshow('Semantic_map', response)

        semantic_map_3c=cv2.cvtColor(response, cv2.COLOR_GRAY2BGR);
        semantic_map_color=cv2.LUT(semantic_map_3c, self.carottes)
        cv2.imshow('Semantic_map_color', semantic_map_color)
        #applying the semantic segmentation map as an overlay on the input
        alpha=0.5
        cv2.addWeighted(semantic_map_color, alpha, self.frame_patch, 1 - alpha, 0, self.frame_patch)
        cv2.imshow('Semantic segmentation overlay', self.frame_patch)
        cv2.waitKey(4)
