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
nb_classes=33
patchSize=256
patchesPerImage=50
allow_display=True
process_labels_histogram=False
nan_management='avoid'

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
dataset_raw_train=DataProvider_input_pipeline.extractFilenames(raw_data_dir_train, '*.png')
dataset_references_train=DataProvider_input_pipeline.extractFilenames(reference_data_dir_train, '*labelIds.png')
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
            input_rgb, cloud_map, img_labels=display_sentinel_image(img, mask_clouds_OPAQUE=True, mask_clouds_CIRRUS=False)
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
    data_provider=DataProvider_input_pipeline.FileListProcessor_Semantic_Segmentation(dataset_raw_train, dataset_references_train,
                                                                shuffle_samples=True,
                                                                patch_ratio_vs_input=patchSize,
                                                                max_patches_per_image=patchesPerImage,
                                                                image_area_coverage_factor=2.0,
                                                                num_preprocess_threads=4,
                                                                apply_random_flip_left_right=True,
                                                                apply_random_flip_up_down=False,
                                                                apply_random_brightness=0.5,
                                                                apply_random_saturation=0.5,
                                                                apply_whitening=True,
                                                                batch_size_train=50,
                                                                use_opencv_imread=False,
                                                                balance_classes_distribution=True,
                                                                classes_entropy_threshold=0.6,
                                                                field_of_view=0)
else:
    print("Testing the TEST pipeline mode")
    data_provider=DataProvider_input_pipeline.FileListProcessor_Semantic_Segmentation(dataset_raw_train, dataset_references_train,
                                                                shuffle_samples=False,
                                                                patch_ratio_vs_input=patchSize,
                                                                max_patches_per_image=patchesPerImage,
                                                                image_area_coverage_factor=1.0,
                                                                num_preprocess_threads=1,
                                                                apply_random_flip_left_right=False,
                                                                apply_random_flip_up_down=False,
                                                                apply_random_brightness=None,
                                                                apply_random_saturation=None,
                                                                apply_whitening=True,
                                                                batch_size_train=1,
                                                                use_opencv_imread=False,
                                                                balance_classes_distribution=False,
                                                                classes_entropy_threshold=None,
                                                                field_of_view=0)
#retreive a single sample (testing)
deepnet_feed=data_provider.deepnet_data_queue.dequeue()

### Visualisation Contours Sobel
with tf.name_scope('semantic_contours'):
    ''' deepnet_feed is a 3D tensor (height, width, 1) but
    the contour detection expects a batch of mono channel images (batch, height, width)
    ==> here it then needs a 3D tensor (1, height, width)
    '''
    reference_crop=tf.squeeze(tf.expand_dims(
                    tf.cast(
                        tf.slice( deepnet_feed,
                            begin=[0,0, data_provider.single_image_raw_depth],
                            size=[-1,-1,data_provider.single_image_reference_depth])
                        ,dtype=tf.int32),0)
                        ,-1)

    print('input batch shape= '+str(reference_crop.get_shape().as_list()))
    contours = DataProvider_input_pipeline.convert_semanticMap_contourMap(reference_crop)
#######################################################################

#create a session with controlled memory allocation (only uses what is required)
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth=True
sess=tf.InteractiveSession(config=session_config)

#summary writer
writer = tf.summary.FileWriter(sessionFolder, sess.graph)

init_op=[tf.global_variables_initializer(), tf.local_variables_initializer()]
sess.run(init_op)
#initialize and feed the filenames string queue
# create coordinated threads to feed the queue
# -> fcreate the coordinator

coord = tf.train.Coordinator()
data_provider.start(session=sess, coordinator=coord)

#run one deep net iteration
try:

  for step in xrange(100000):
      #stop condition
      if coord.should_stop():
          break
      #training step case
      result, contours_disp=sess.run([deepnet_feed, contours ])
      print('One iteration, output=')
      print(result[:,:,:3].shape)
      input_crop=result[:,:,:3]
      sample_minVal=np.min(input_crop)
      sample_maxVal=np.max(input_crop)
      print('Sample value range (min, max)=({minVal}, {maxVal})'.format(minVal=sample_minVal, maxVal=sample_maxVal))
      input_crop_norm=(input_crop-sample_minVal)*255.0/(sample_maxVal-sample_minVal)
      if allow_display is True:
          cv2.imshow('input crop', cv2.cvtColor(input_crop_norm.astype(np.uint8), cv2.COLOR_RGB2BGR))
          cv2.imshow('reference crop', result[:,:,3].astype(np.uint8)*int(255/nb_classes))
          cv2.imshow('reference contours_disp', contours_disp[0].astype(np.uint8)*255)
          cv2.waitKey(1000)

except Exception, e:
    # Report exceptions to the coordinator.
    coord.request_stop(e)
finally:
    # Terminate as usual.  It is innocuous to request stop twice.
    coord.request_stop()
    coord.join(data_provider.enqueue_threads)
sess.close()

print('######## Stopped process at step '+str(step))

if process_labels_histogram is True:
    nb_pix=np.sum(class_count)
    plt.plot(class_count/nb_pix)
    plt.title('Class probabilities, nb_pixels='+str(nb_pix))
    plt.savefig('RS_dataset_Class probabilities.eps')
    #DataProvider_input_pipeline.plot_sample_channel_histograms(result, filenameID="last_crop_")
    if allow_display is True:
        plt.show()
