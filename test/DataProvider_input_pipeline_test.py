#script aiming at testing FileListProcessor_input_test
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import datetime
import numpy as np
import argparse

import DataProvider_input_pipeline

workingFolder='test_datapipeline'
sessionFolder=os.path.join(workingFolder, datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
os.makedirs(sessionFolder)
os.chdir(sessionFolder)

'''first focus on a set of folders with raw ad reference data,
here all data parameters are hard written since all data is supposed to be versionned
'''
raw_data_dir_train = "../../datasamples/semantic_segmentation/raw_data/"
reference_data_dir_train = "../../datasamples/semantic_segmentation/labels/"
#raw_data_dir_train ="/home/alben/workspace/Datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/"
#reference_data_dir_train = "/home/alben/workspace/Datasets/CityScapes/gtFine_trainvaltest/gtFine/train/"
nb_classes=33
patchSize=256
patchesPerImage=1000
allow_display=True
process_labels_histogram=False
nan_management='zeros'

parser = argparse.ArgumentParser(description='DataProvider_input_pipeline_test')
parser.add_argument('--avoid-display', dest='avoid_display', action='store_true',
                    help='avoid display in a Xwindow to let the scrip work on non X systems')
parser.add_argument('--check-data', dest='check_data', action='store_true',
                    help='If True/1, then iteratively tries to read each image of the dataset to ensure readability')
parser.add_argument('--mode-test', dest='mode_test', action='store_true',
                    help='Switch from Training mode (random crops by default) to Testing mode (adjacent cropping with overlap)')
parser.add_argument('--mode-labels-count', dest='labels_count', action='store_true',
                    help='compute class histogram over all the dataset')
parser.add_argument('--data-folder', dest='data_folder', default='',
                    help='change the data folder from default svn versionned image samples to a specific directory')
parser.add_argument('--field_of_view', dest='field_of_view', default=0,
                    help='specify the boundary width reduction on the reference data to remove')
parser.add_argument('--write_tfRecords', dest='write_tfRecords', action='store_true',
                    help='activate tf records writing, to be loaded at full speed for the next training session')
parser.add_argument('--tfRecords_name', dest='tfRecords_name', default='/tmp/myDataset',
                    help='activate tf records writing, to be loaded at full speed for the next training session')

processCommands = parser.parse_args()
field_of_view=processCommands.field_of_view
if processCommands.data_folder != '':
    print('Testing of specific data folder : '+ processCommands.data_folder)
    raw_data_dir_train=processCommands.data_folder
if processCommands.avoid_display:
    allow_display = False
if processCommands.labels_count:
    process_labels_histogram = True

#get list of filenames
dataset_raw_files=DataProvider_input_pipeline.extractFilenames(raw_data_dir_train, '*.png')
dataset_references_files=DataProvider_input_pipeline.extractFilenames(reference_data_dir_train, '*labelIds.png')

if processCommands.check_data:
    print('Trying to load each image of the train dataset, this may take some time...')
    nb_channels = cv2.imread(dataset_raw_files[0], cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYDEPTH).shape[-1]
    print('--> First image number of channels='+str(nb_channels))

    for imageIdx, imageName in enumerate(dataset_raw_files):
        print('Testing image : '+imageName)
        img=cv2.imread(imageName, cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYDEPTH)
        if nb_channels != img.shape[-1]:
            raise ValueError('Error, the number of channels differs with image : ' +str(imageName))
        if img is None:
            raw_input('Failed to read image '+imageName)
            os.rename(imageName, imageName+'.unreadable')
        if not(np.isfinite(img)).any():
            raw_input('Nan or inf found in file to read image '+imageName)
            os.rename(imageName, imageName+'.unreadable')

        ''' Display images section '''
        if allow_display is True:
            #print('Labels (min, max)='+str((img_labels.min(), img_labels.max())))
            input_rgb, cloud_map, img_labels=display_sentinel_image(img, mask_clouds_OPAQUE=True, mask_clouds_CIRRUS=False)
            img_labels[np.where(cloud_map)]=23
            cv2.imshow('input image', DataProvider_input_pipeline.scaleImg_0_255(input_rgb).astype(np.uint8))
            cv2.imshow('reference image', img[:,:,-1].astype(np.uint8)*int(255/nb_classes))
            cv2.imshow('cloud image', cloud_map.astype(np.uint8)*255)
            cv2.waitKey()
            #DataProvider_input_pipeline.plot_sample_channel_histograms(img, filenameID="image_"+str(imageIdx)+'_')
            #plt.show()

###############################
#prepare dataset input pipeline
isTraining=not(processCommands.mode_test)
def apply_pixel_transforms(isTraining):
  if isTraining:
      return 0.5
  else:
      return None

data_provider=DataProvider_input_pipeline.FileListProcessor_Semantic_Segmentation(dataset_raw_files, dataset_references_files,
          nbEpoch=1,
          shuffle_samples=isTraining,
          patch_ratio_vs_input=patchSize,
          max_patches_per_image=patchesPerImage,
          image_area_coverage_factor=int(isTraining)+1.0,#factor 2 on training, 1 on testing
          num_reader_threads=10,#4 threads on training, 1 on testing
          apply_random_flip_left_right=isTraining,
          apply_random_flip_up_down=False,
          apply_random_brightness=apply_pixel_transforms(isTraining),
          apply_random_saturation=apply_pixel_transforms(isTraining),
          apply_whitening=True,
          batch_size=1,
          use_alternative_imread='opencv',
          balance_classes_distribution=isTraining,
          classes_entropy_threshold=0.6,
          opencv_read_flags=cv2.IMREAD_UNCHANGED,
          field_of_view=field_of_view,
          manage_nan_values=nan_management,
          additionnal_filters=None,
          crops_postprocess=None)


### semantic contours extraction
@tf.function
def get_semantic_contours(reference_crop):
    ''' deepnet_feed is a 3D tensor (height, width, 1) but
    the contour detection expects a batch of mono channel images (batch, height, width)
    ==> here it then needs a 3D tensor (1, height, width)
    '''

    print('input batch shape= '+str(reference_crop.get_shape().as_list()))
    formatted_reference_crop=tf.squeeze(tf.cast(reference_crop, tf.int32), -1)
    contours = DataProvider_input_pipeline.convert_semanticMap_contourMap(formatted_reference_crop)
    return contours

#######################################################################
### Optionnal samples recording setup (check here:https://www.tensorflow.org/api_docs/python/tf/io/TFRecordWriter?version=nightly)
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a dictionary with features that may be relevant.
def image_example(image_tensor):#, label):
  image_string=tf.io.serialize_tensor(image_tensor)
  image_shape = image_tensor.shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      #'label': _int64_feature(label),
      #'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

if processCommands.write_tfRecords:
  #setup
  ''' dataset recorder, display loop below is then ran AFTER full dataset recording :
  writer = tf.data.experimental.TFRecordWriter(file_out)
  writer.write(serialized_features_dataset)
  '''
  file_out = "%s.tfrecords" % processCommands.tfRecords_name
  print('Samples will be written as tfrecords in files: ', file_out)
  writer= tf.io.TFRecordWriter(file_out)

#######################################################################
### Samples extraction loop
for step, batch in enumerate(data_provider.dataset):
  print('====== New step='+str(step))#, 'batch=',batch[0])
  sample=batch[0]
  if processCommands.write_tfRecords:
    sample_proto=image_example(sample)
    print('Serialized sample:', sample_proto)
    writer.write(sample_proto.SerializeToString())
  if allow_display is True:
    input_crop=sample[:,:,:3].numpy()
    reference =sample[:,:,3:].numpy()
    contours=get_semantic_contours(tf.expand_dims(sample[:,:,3:],0)).numpy().squeeze(0)
    print('input_crop shape =', input_crop.shape)
    print('reference_crop shape =', reference.shape)
    print('contours_crop shape =', contours.shape)
    sample_minVal=np.min(input_crop)
    sample_maxVal=np.max(input_crop)
    print('Sample value range (min, max)=({minVal}, {maxVal})'.format(minVal=sample_minVal, maxVal=sample_maxVal))
    input_crop_norm=(input_crop-sample_minVal)*255.0/(sample_maxVal-sample_minVal)
    cv2.imshow('input crop', cv2.cvtColor(input_crop_norm.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imshow('reference crop', reference.astype(np.uint8)*int(255/nb_classes))
    cv2.imshow('reference contours_disp', contours.astype(np.uint8)*255)
    cv2.waitKey(1000)

if processCommands.write_tfRecords:
  writer.close()
#loop ended, final pause before closing
if allow_display is True:
  print('finished crop sample display, press a key to stop from an active opencv image show window')
  cv2.waitKey()

print('######## Stopped process at step '+str(step))
cv2.waitKey()

if process_labels_histogram is True:
    nb_pix=np.sum(class_count)
    plt.plot(class_count/nb_pix)
    plt.title('Class probabilities, nb_pixels='+str(nb_pix))
    plt.savefig('RS_dataset_Class probabilities.eps')
    #DataProvider_input_pipeline.plot_sample_channel_histograms(result, filenameID="last_crop_")
    if allow_display is True:
        plt.show()
