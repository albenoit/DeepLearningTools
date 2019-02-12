# python 2&3 compatibility management
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

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
raw_data_dir_train = "/home/alben/workspace/DeepLearningRessources_july2017/trunk/TensorFlow/datasamples/EarthEngine/landcover_semantic_segmentation_with_clouds"#{}home/alben/workspace/DeepLearningRessources/trunk/TensorFlow/listic-deeptool/datasamples/IGARSS2017/"
nb_classes=23+1#23 landcover classes + the clouds class
patchSize=224
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

processCommands = parser.parse_args()
if processCommands.data_folder != '':
    print('Testing of specific data folder : '+ processCommands.data_folder)
    raw_data_dir_train=processCommands.data_folder
if processCommands.avoid_display:
    allow_display = False
if processCommands.labels_count:
    process_labels_histogram = True

#get list of filenames
print('Looking for files in '+str(raw_data_dir_train) +' from working folder '+os.getcwd())
dataset_raw_train=DataProvider_input_pipeline.extractFilenames(raw_data_dir_train, '*.tif')

def display_sentinel_image(input_img, mask_clouds_OPAQUE=True, mask_clouds_CIRRUS=True):
        input_rgb=input_img[:,:,1:4]
        if mask_clouds_OPAQUE == True and mask_clouds_CIRRUS == True:
            cloud_map=input_img[:,:,15]>=1024 #Q60 band greater or equal to 1024
        elif mask_clouds_CIRRUS is True:
            cloud_map=input_img[:,:,15]>1024 #Q60 band greater than 1024
        elif mask_clouds_OPAQUE is True:
            cloud_map=input_img[:,:,15]==1024 #Q60 band equal to 1024
        #print('Cloud pixels='+str(np.sum(cloud_map.astype(int))))
        img_labels=input_img[:,:,17]
        print('Cloud pixels='+str(np.sum(cloud_map)))
        return input_rgb, cloud_map, img_labels

if processCommands.check_data:
    print('Trying to load each image of the train dataset, this may take some time...')
    nb_channels = cv2.imread(dataset_raw_train[0], cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYDEPTH).shape[-1]
    print('--> First image number of channels='+str(nb_channels))

    for imageIdx, imageName in enumerate(dataset_raw_train):
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
            input_rgb, cloud_map, img_labels=display_sentinel_image(img, mask_clouds_OPAQUE=True, mask_clouds_CIRRUS=True)
            img_labels[np.where(cloud_map)]=23
            cv2.imshow('input image', DataProvider_input_pipeline.scaleImg_0_255(input_rgb).astype(np.uint8))
            cv2.imshow('reference image', img[:,:,-1].astype(np.uint8)*int(255/nb_classes))
            cv2.imshow('cloud image', cloud_map.astype(np.uint8)*255)
            cv2.waitKey()
            #DataProvider_input_pipeline.plot_sample_channel_histograms(img, filenameID="image_"+str(imageIdx)+'_')
            #plt.show()

#init the input pipeline
if not(processCommands.mode_test):
    print("Testing the TRAIN pipeline mode")
    data_provider=DataProvider_input_pipeline.FileListProcessor_Semantic_Segmentation(dataset_raw_train, None,
		                                                    shuffle_samples=True,
                                                        nbEpoch=1,
                                                        patch_ratio_vs_input=patchSize,
		                                                    max_patches_per_image=patchesPerImage,
		                                                    image_area_coverage_factor=2.0,
		                                                    num_preprocess_threads=1,
		                                                    apply_random_flip_left_right=True,
		                                                    apply_random_flip_up_down=True,
		                                                    apply_random_brightness=None,
		                                                    apply_random_saturation=None,
		                                                    apply_whitening=False,
		                                                    batch_size=1,
		                                                    use_alternative_imread='gdal',
		                                                    balance_classes_distribution=True,
		                                                    classes_entropy_threshold=0.3,
		                                                    opencv_read_flags=cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYDEPTH,
		                                                    field_of_view=0,
                                                            manage_nan_values=nan_management)
else:
    print("Testing the TEST pipeline mode")
    data_provider=DataProvider_input_pipeline.FileListProcessor_Semantic_Segmentation(dataset_raw_train, None,
		                                                    shuffle_samples=False,
                                                        nbEpoch=1,
		                                                    patch_ratio_vs_input=patchSize,
		                                                    max_patches_per_image=1,
		                                                    image_area_coverage_factor=1.0,
		                                                    num_preprocess_threads=1,
		                                                    apply_random_flip_left_right=False,
		                                                    apply_random_flip_up_down=False,
		                                                    apply_random_brightness=None,
		                                                    apply_random_saturation=None,
		                                                    apply_whitening=False,
		                                                    batch_size=1,
		                                                    use_alternative_imread='gdal',
		                                                    balance_classes_distribution=False,
		                                                    classes_entropy_threshold=0.3,
		                                                    opencv_read_flags=cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYDEPTH,
                                                        field_of_view=0,
                                                        manage_nan_values=nan_management)

#retreive a single sample (testing)
deepnet_feed=tf.squeeze(data_provider.dataset_iterator.get_next(),0)

#create a session with controlled memory allocation (only uses what is required)
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth=True
sess=tf.InteractiveSession(config=session_config)

#summary writer
writer = tf.summary.FileWriter(sessionFolder, sess.graph)

init_op=[tf.global_variables_initializer(), tf.local_variables_initializer(), data_provider.getIteratorInitializer()]
sess.run(init_op)
#initialize and feed the filenames string queue
# create coordinated threads to feed the queue
# -> fcreate the coordinator

coord = tf.train.Coordinator()

class_count=np.zeros(nb_classes)
#run one deep net iteration
try:

  for step in six.moves.range(10):
      #stop condition
      if coord.should_stop():
          break
      #training step case
      result=sess.run(deepnet_feed)
      print('#### step='+str(step))
      print('-> output shape='+str(result.shape))
      input_crop, cloud_map, img_labels=display_sentinel_image(result, mask_clouds_OPAQUE=True, mask_clouds_CIRRUS=False)
      #print('Labels (min, max)='+str((img_labels.min(), img_labels.max())))
      img_labels[np.where(cloud_map)]=23
      if process_labels_histogram is True:

          img_class_count, img_labels_hist=np.histogram(img_labels, bins=nb_classes, range=(0,nb_classes-1))
          print('Image classes count='+str(img_class_count))
          class_count+=img_class_count
      if process_labels_histogram is False:
          if allow_display is True:
              cv2.imshow('input crop, step='+str(step), DataProvider_input_pipeline.scaleImg_0_255(input_crop).astype(np.uint8))
              cv2.imshow('reference crop, step='+str(step), result[:,:,-1].astype(np.uint8)*int(255/nb_classes))
              cv2.imshow('cloud map, step='+str(step), cloud_map.astype(np.uint8)*255)
              cv2.waitKey(1000)
          else:
             cv2.imwrite('input_crop_'+str(step)+'.bmp', DataProvider_input_pipeline.scaleImg_0_255(input_crop).astype(np.uint8))
             cv2.imwrite('reference_crop_'+str(step)+'.bmp', result[:,:,-1].astype(np.uint8)*int(255/nb_classes))
  #loop ended, final pause before closing
  if allow_display is True:
    print('finished crop sample display, press a key to stop from an active opencv image show window')
    cv2.waitKey()

except Exception as e:
    print('Exception received:'+str(e))
    # Report exceptions to the coordinator.
    coord.request_stop(e)
finally:
    # Terminate as usual.  It is innocuous to request stop twice.
    coord.request_stop()
sess.close()

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
