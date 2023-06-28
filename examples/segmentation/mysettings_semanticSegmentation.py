'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
==> application : semantic segmentation on the cats and dogs Oxford III Pets dataset

FULL PROCESS USE EXAMPLE:
1. TRAIN/VAL : start a train/val session using command (a singularity container with an optimized version of Tensorflow is used here):
singularity run --nv /home/alben/install/nvidia/tf2_addons.sif experiments_manager.py --usersettings=examples/segmentation/mysettings_semanticSegmentation.py

2. SERVE MODEL : start a tensorflow model server on the produced eperiment models using command (the -psi command permits to start tensorflow model server installed in a singularity container):

3. REQUEST MODEL : start a client that sends continuous requests to the server

Check training logs : apptainer exec --nv /path/to/tf2_addons.sif tensorboard --logdir experiments/examples/semantic_segmentation
'''

import deeplearningtools.DataProvider_input_pipeline
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import cv2 #for ClientIO only

#-> set here your own working folder
workingFolder='experiments/examples/semantic_segmentation'

#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='Oxford-IIIT-Pets'

''' define here some hyperparameters to adjust the experiment
===> Note that this dictionnary will complete the session name
'''
hparams={'learningRate':0.001,
         'nbClasses':34,#set the number of classes in the considered dataset
         'smoothedParams':True,
         'nbEpoch':20,
         'batchSize':4,
         'nbClasses':3,
         'patchSize':128,
        }

''''set the list of GPUs involved in the process. HOWTO:
->if using CPU only mode, let an empty list
->if using a single GPU, only the first ID of the list will be considered
->if using multiple GPUs, each GPU ID will be considered
=> general recommendation: always try to focus on unused GPUs to avoid conflicts
with other processing jobs, yours and the ones of your colleagues.
Then, connect to the processing node and type in command line 'nvidia-smi'
to check which gpu is free (very few used memory and GPU )
'''
used_gpu_IDs=[]
#set here XLA optimisation flags, either tf.OptimizerOptions.OFF#ON_1#OFF
useXLA=True

#profile some training steps to check pipeline processing time bottlenecks (from Tensorboard)
use_profiling=True

#-> define here the used model under variable 'model'
model_file='examples/segmentation/model_mobilenetV2_Unet.py'

# activate weight moving averaging over itarations (Polyak-Ruppert)
weights_moving_averages=False

# random seed used to init weights, etc. Use an integer value to make experiments reproducible
random_seed=42


#semantic segmentation task image patches config
field_of_view=109
test_patch_overlapping_ratio=0.75 #-> patch overlapping when evaluating/predicting
patchSize=128
server_patch_size=128
server_crops_per_batch=4


# stop condition, taking into account if val_loss does not decrease for early_stopping_patience epoch
nbEpoch=hparams['nbEpoch']
early_stopping_patience=10

#set here paths to your data used for train, val
#=> download tensorflow dataset for demo purpose
import tensorflow_datasets as tfds
dataset, info = tfds.load('oxford_iiit_pet:3.2.0', with_info=True)
print('Dataset info :', info)
nb_train_images=info.splits['train'].num_examples#len(DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_dir_train_, file_extension=raw_data_filename_extension, raiseOnEmpty=False))
nb_val_images=info.splits['test'].num_examples#len(DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_dir_val_, file_extension=raw_data_filename_extension, raiseOnEmpty=False))

raw_data_dir_train_ = ""#"/home/alben/workspace/Datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/"
reference_data_dir_train_ = ""#"/home/alben/workspace/Datasets/CityScapes/gtFine_trainvaltest/gtFine/train/"
raw_data_dir_val_ = ""#"/home/alben/workspace/Datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/val/"
reference_data_dir_val_ = ""#"/home/alben/workspace/Datasets/CityScapes/gtFine_trainvaltest/gtFine/val/"
raw_data_filename_extension='*.png'
ref_data_filename_extension='*labelIds.png'
raw_data_dir_train=(raw_data_dir_train_, reference_data_dir_train_)
raw_data_dir_val=(raw_data_dir_val_, reference_data_dir_val_)

number_of_crops_per_image=100
nb_train_samples=nb_train_images*number_of_crops_per_image# number of images * number of crops per image
nb_val_samples=nb_val_images*number_of_crops_per_image
VAL_SUBSPLITS = 5
batch_size=hparams['batchSize']
steps_per_epoch=nb_train_samples//batch_size
validation_steps=nb_val_samples//batch_size//VAL_SUBSPLITS
reference_labels=['semantic_labels']

########## MODEL SERVING/PRODUCTION PARAMETERS SECTION ################
#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=1#timeout limit when a client requests a served model
serve_on_gpu=True #uncomment to activate model serving on GPU instead of CPU
served_input_names=['input']
served_head_names=['semantic_labels', 'logits']

########## LOCAL PARAMETERS (ONLY USED BELOW) SECTION ################

########## TRAIN/VAL PERSONNALIZED FUNCTIONS SECTION ################

# add here any additionnal callback to use along the train/val process
def addon_callbacks(model, train_samples, val_samples):
  ''' optionnal callbacks can be defined here
  Arg: the defined model
  Returns a list of tf.keras.callbacks or an empty list
  '''
  # Note this link to add pr_curves : https://medium.com/@akionakas/precision-recall-curve-with-keras-cd92647685e1
  return []

def get_learningRate():
  ''' define here the learning rate
  Returns a sclalar (float) or a scheduler
  '''
  return hparams['learningRate']

def get_optimizer(model, loss, learning_rate):
    '''define here the specific optimizer to be used
    Returns a tensorflow optimizer object
    '''
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return optimizer

def get_metrics(model, loss):

  #define a custom Mean IoU metric, taking into the fact that predictions are per class logits
  class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(MeanIoU, self).update_state(y_true, y_pred, sample_weight)

  return ['accuracy', MeanIoU(num_classes=hparams['nbClasses'])]

def get_total_loss(model):
    '''a specific loss can be defined here or simply use a string that refers to a keras loss
    Args:
        model: the model to be optimized that may be used to focus loss on a set of specific layers or so
    Returns:
        a keras implemented loss represented by a string or a custom loss
        => it is recommended to return a tensor named 'loss' in order to enable some
        useful default options such as early stopping
    '''


    return 'sparse_categorical_crossentropy'

'''
Define here the input pipelines : a common function for train and validation modes
'''
def get_input_pipeline(raw_data_files_folder, isTraining, batch_size, nbEpoch):
  ''' define an input pipeline a basic example here:
  -> load a standard dataset with tuples (image, label)
  TODO, look at the doc here : https://www.tensorflow.org/programmers_guide/datasets
  @param raw_data_files_folder : the variable that could target a dataset/folder...
  @param isTraining : a boolean that activates batch shuffling
  '''
  def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

  if isTraining:
    @tf.function
    def load_image_train(datapoint):
      input_image = tf.image.resize(datapoint['image'], (128, 128))
      input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

      if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

      input_image, input_mask = normalize(input_image, input_mask)

      return input_image, input_mask
    target_dataset = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    target_dataset=target_dataset.cache().shuffle(100).batch(batch_size).repeat()
    target_dataset=target_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  else:
    def load_image_test(datapoint):
      input_image = tf.image.resize(datapoint['image'], (patchSize, patchSize))
      input_mask = tf.image.resize(datapoint['segmentation_mask'], (patchSize, patchSize))

      input_image, input_mask = normalize(input_image, input_mask)

      return input_image, input_mask
    target_dataset = dataset['test'].map(load_image_test).batch(batch_size)

  return target_dataset

'''
################################################################################
## Serving (production) section, define here :
-get_served_module():  define how the model will be applied on production data, applying custom pre and post processing
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
'''
serving_img_shape=[server_crops_per_batch, server_patch_size,server_patch_size,3]
def get_served_module(model, model_name):
  ''' following https://www.tensorflow.org/guide/saved_model
      Create a custom module to specify how the model will be used in production/serving
      specific preprocessing can be defined as well as post-processing
  '''
  class ExportedModule(tf.Module):
    def __init__(self, model):
      super().__init__()
      self.model=model

    @tf.function(input_signature=[tf.TensorSpec(shape=serving_img_shape, dtype=tf.uint8, name=served_input_names[0])])
    def served_model(self, input):
      ''' a decorated function that specifies the input data format, processing and output dict
        Args: input tensor(s)
        Returns a dictionnary of {'output key':tensor}
      '''
      def normalize(input_image):
        return input_image/ 255.0

      input_standardized_casted=tf.map_fn(normalize, tf.cast(input, tf.float32))
      logits=model(input_standardized_casted)

      return {served_head_names[0]:tf.saturate_cast(tf.argmax(logits,3, name='argmax_image'), tf.uint8),
              served_head_names[1]:logits}
  return ExportedModule(model)

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
        for l in range(self.nb_crops_lines):
          for c in range(self.nb_crops_colums):
            self.crops_positions_in.append((l*self.patch_effective_width,c*self.patch_effective_width))
        #adding bottom and right borders
        for l in range(self.nb_crops_lines):
          self.crops_positions_in.append((l*self.patch_effective_width,self.outframe.shape[1]-server_patch_size))
        for c in range(self.nb_crops_colums):
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
            Label(  'background'            ,  0 ,      0 , 'background'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'pet_boundaries'        ,  1 ,      1 , 'ambiguous'            , 1       , False        , False        , (244, 35,232) ),
            Label(  'pet_body'              ,  2 ,      2 , 'foreground'           , 2       , True         , False        , (220, 20, 60) ),
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
        for i in range(server_crops_per_batch):
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
          for idx in range(server_crops_per_batch):
            cv2.imshow('input crop'+str(idx), self.frame_patch[idx,:,:,:])
          cv2.waitKey(5)

        if self.debugMode is True:
            print('Input frame'+str(self.frame_patch.shape))
        return {served_input_names[0]:self.frame_patch}

    def decodeResponse(self, result):
        ''' receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        '''
        response = np.reshape(np.array(result.outputs[served_head_names[0]].int_val),serving_img_shape[:-1]).astype(np.uint8)

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
