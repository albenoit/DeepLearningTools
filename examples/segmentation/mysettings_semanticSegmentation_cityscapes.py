'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
'''

import DataProvider_input_pipeline
import helpers.loss as helpers_loss
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import cv2 #for ClientIO only

#-> set here your own working folder
workingFolder='experiments/semantic_segmentation'

#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='Cityscapes'

''' define here some hyperparameters to adjust the experiment
===> Note that this dictionnary will complete the session name
'''
hparams={'learningRate':0.001,
         'nbClasses':34,#set the number of classes in the considered dataset
         'smoothedParams':True,
         'nbEpoch':20,
         'batchSize':32,
         'outChannels':34,
         'patchSize':224
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
used_gpu_IDs=[0]
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
field_of_view=0
test_patch_overlapping_ratio=0.75 #-> patch overlapping when evaluating/predicting
patchSize=hparams['patchSize']
server_patch_size=hparams['patchSize']
server_crops_per_batch=4


# stop condition, taking into account if val_loss does not decrease for early_stopping_patience epoch
nbEpoch=hparams['nbEpoch']
early_stopping_patience=10

#set here paths to your data used for train, val
raw_data_dir_train_ = "/home/alben/workspace/Datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/"
reference_data_dir_train_ = "/home/alben/workspace/Datasets/CityScapes/gtFine_trainvaltest/gtFine/train/"
raw_data_dir_val_ = "/home/alben/workspace/Datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/val/"
reference_data_dir_val_ = "/home/alben/workspace/Datasets/CityScapes/gtFine_trainvaltest/gtFine/val/"
raw_data_filename_extension='*.png'
ref_data_filename_extension='*labelIds.png'
raw_data_dir_train=(raw_data_dir_train_, reference_data_dir_train_)
raw_data_dir_val=(raw_data_dir_val_, reference_data_dir_val_)
nb_train_images=5000
nb_val_images=300
number_of_crops_per_image=100
nb_train_samples=nb_train_images*number_of_crops_per_image# number of images * number of crops per image
nb_val_samples=nb_val_images*number_of_crops_per_image
batch_size=hparams['batchSize']
steps_per_epoch=nb_train_samples//batch_size
validation_steps=nb_val_samples//batch_size
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
  #FIXME check/monitor :
  #-> https://github.com/keras-team/keras/issues/13283
  #-> https://github.com/tensorflow/tensorflow/issues/28868
  def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    with val_samples.take(batch_size) as val_batch:

      print(val_batch)
      print(val_batch[0], ref_batch[1])
      tval_pred_raw = model.predict(test_pred_raw)
      val_pred = np.argmax(val_pred_raw, axis=1)

    # Calculate the confusion matrix.
    cm = sklearn.metrics.confusion_matrix(val_ref, val_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
      tf.summary.image("Confusion Matrix", cm_image, step=epoch)

  # Define the per-epoch callback.
  #cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

  file_writer_img = tf.summary.create_file_writer('logcm')

  def log_image(epoch, logs):
    img_cnn = model.outputs[0]#.output
    print('img_cnn',img_cnn)
    # Log the cnn image as an image summary.
    with file_writer_img.as_default():
      tf.summary.image(name="my_conv2d",data=img_cnn , step=epoch)

  # Define the per-epoch callback.
  img_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_image)

  return [img_callback]

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

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

  return ['accuracy', MeanIoU(num_classes=hparams['outChannels'])]

def get_total_loss(model):
    '''a specific loss can be defined here or simply use a string that refers to a keras loss
    Args:
        model: the model to be optimized that may be used to focus loss on a set of specific layers or so
    Returns:
        a keras implemented loss represented by a string or a custom loss
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
  raw_data_files=DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_files_folder[0], file_extension=raw_data_filename_extension)
  dataset_references_train=DataProvider_input_pipeline.extractFilenames(raw_data_files_folder[1], ref_data_filename_extension)
  nb_raw_images=len(raw_data_files)
  nb_ref_images=len(dataset_references_train)
  print('Input files found (raw, ref)='+str((nb_raw_images, nb_ref_images)))
  if nb_raw_images!=nb_ref_images:
      raise ValueError('Raw images and reference image numbers differ, check datasets')

  # set patches boundaries overlapping. Should be related to model field of view.
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
  def apply_pixel_transforms(isTraining):
      if isTraining:
          return 0.5
      else:
          return None
  #init the input pipeline
  dataset_nbEpoch=nbEpoch#for the training dataset
  if isTraining is False:
    dataset_nbEpoch=1

  def crops_postprocess(sample):
    ''' a helper function that post processes the crops generated by the basic dataprovider to more specific data:
        -> the data provider original crops are a set of channels, each of the same size (Raw cahnnels+optionnal reference channel)
        -> this function zips the original raw channels with the cropped reference, avoiding the borders impacted by
        the model field of view
    '''
    raw_image=tf.slice( sample,
                                  begin=[0,0,0],
                                  size=[-1,-1,3])
    with tf.name_scope('prepare_reference_data'):
        #-> get reference data restricted to the center part of the images
        reference_crop=tf.cast(
                                tf.slice( sample,
                                    begin=[field_of_view//2, field_of_view//2, 3],
                                    size=[patchSize-field_of_view, patchSize-field_of_view,1])
                            ,dtype=tf.int32)
    return (raw_image, reference_crop)

  data_provider=DataProvider_input_pipeline.FileListProcessor_Semantic_Segmentation(raw_data_files, dataset_references_train,
          nbEpoch=dataset_nbEpoch,
          shuffle_samples=isTraining,
          patch_ratio_vs_input=patchSize,
          max_patches_per_image=number_of_crops_per_image,
          image_area_coverage_factor=int(isTraining)+1.0,#factor 2 on training, 1 on testing
          num_reader_threads=4,#4 threads on training, 1 on testing
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
          manage_nan_values='avoid',
          crops_postprocess=crops_postprocess)

  return data_provider.dataset

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

    @tf.function(input_signature=[tf.TensorSpec(shape=serving_img_shape, dtype=tf.uint8)])
    def served_model(self, input):
      ''' a decorated function that specifies the input data format, processing and output dict
        Args: input tensor(s)
        Returns a dictionnary of {'output key':tensor}
      '''
      input_standardized_casted=tf.map_fn(tf.image.per_image_standardization, tf.cast(input, dtype=tf.float32))
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
        field_of_view = int(np.ceil(test_patch_overlapping_ratio*server_patch_size) // 2 * 2 + 1)

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
        return self.frame_patch

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
