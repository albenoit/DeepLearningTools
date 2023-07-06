# ========================================
# FileName: tfrecords_io.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of functions to write and read tfrecords datasets, directly run this script to test write and read of a dataset of 2 sklearn images
# for DeepLearningTools.
# =========================================

import tensorflow as tf
from deeplearningtools.helpers import tensor_msg_io
import numpy as np
import cv2

def image_tfrecords_dataset(filename, hasLabels=False):
  '''
  Assuming a set of tfrecords file is pointed by filename, ex:'images.tfrecords',
  create a data provider that loads them for training/testing models

  
  :param filename: A path to the tfrecord files.
  :param haslabel: A boolean, false by default that specifies if an iteger label is expected or not

  :returns: A tf.data.Dataset WITHOUT PREFETCH NOR BATCH, specify your own
  '''

  raw_image_dataset = tf.data.TFRecordDataset(filename)

  # Create a dictionary describing the features.
  image_feature_description = {
      'image_raw': tf.io.FixedLenFeature([], tf.string),  # image is supposed to be encoded as a serialized tensor
  }
  if hasLabels:
      image_feature_description.update({'label': tf.io.FixedLenFeature([], tf.int64)})

  def _parse_image_function(example_proto):
      # Parse the input tf.Example proto using the dictionary above.
      print('example proto', example_proto)
      flat_sample = tf.io.parse_single_example(example_proto, image_feature_description)
      print('example with serialized tensor', flat_sample)
      sample = tf.io.parse_tensor(flat_sample['image_raw'], tf.uint8)  # serialized tensor is supposed to be of type uint8 in our example
      return sample

  return raw_image_dataset.map(_parse_image_function)

def display_image_tfrecords_dataset(filename='test_dataset.tfrecords'):
  """
  Load a dataset from a TFRecords file and display the recorded samples.

  Suppose a dataset pointed by files 'test_dataset.tfrecords' exists, load it and display the recorded samples_saving_queuet

  :param filename: Path to the TFRecords file containing the dataset. (default: 'test_dataset.tfrecords')
  :type filename: str

  :return: None
  """
  #Read the created dataset
  dataset = image_tfrecords_dataset(filename)
  for sample in dataset:
    #print('sample shape', sample.shape)
    image_raw = sample.numpy()
    reference=None
    print('image_raw.shape[-1]',image_raw.shape[-1])
    if image_raw.shape[-1]==4:
      print('RGB image + reference channel')
      input_crop=image_raw[:,:,:3]
      reference=image_raw[:,:,3]
    elif image_raw.shape[-1]==3:
      print('Single RGB image ')
      input_crop=image_raw
    elif image_raw.shape[-1]==2:
      print('Gray image + reference channel')
      input_crop=image_raw[:,:,0]
      reference=image_raw[:,:,1]
    elif image_raw.shape[-1]==1 or len(image_raw.shape)==2:
      print('Single gray image')
      input_crop=image_raw
    else:
      raise ValueError('Failed to display array of shape '+str(image_raw.shape))
    #display relying on OpenCV
    sample_minVal=np.min(input_crop)
    sample_maxVal=np.max(input_crop)
    print('Sample value range (min, max)=({minVal}, {maxVal})'.format(minVal=sample_minVal, maxVal=sample_maxVal))
    input_crop_norm=(input_crop-sample_minVal)*255.0/(sample_maxVal-sample_minVal)
    cv2.imshow('TEST input crop rescaled (0-255)', cv2.cvtColor(input_crop_norm.astype(np.uint8), cv2.COLOR_RGB2BGR))
    if reference is not None:
      cv2.imshow('TEST reference crop (classID*20)', reference.astype(np.uint8)*20)
    print('when opencv image display window is active, press a key to continue...')
    cv2.waitKey()


def write_image_dataset_no_display(images_dataset, file_out="demo_image_dataset.tfrecords"):
  """
  Add serialization node to the dataprovider and write the dataset directly.

  :param images_dataset: The image dataset to be written.
  :type images_dataset: tf.data.Dataset

  :param file_out: The output path for the TFRecords file. (default: "demo_image_dataset.tfrecords")
  :type file_out: str

  :return: None
  """
  def serialize_sample(sample):
    """
    Takes as input a tensor, transform to protobuffer and returns it serialized
    """
    return tf.py_function(tensor_msg_io.serialize_image_float_example, [sample[0]], tf.string)

  writer = tf.io.TFRecordWriter(file_out)
  print('Writing dataset without display...can be long...')
  for sample in  images_dataset:
    print('sample', sample)
    writer.write(tensor_msg_io.serialize_image_float_example(sample))#images_dataset.map(serialize_sample))
  print('Dataset writing done !')
  writer.close()

if __name__ == "__main__":
	#test code
	#-> get demo images
	from sklearn.datasets import load_sample_images
	dataset = load_sample_images()     
	print('demo images', dataset.images)
	images_tf=tf.constant(dataset.images, dtype=tf.uint8)
	print('images_tf', images_tf)
	demo_dataset = tf.data.Dataset.from_tensor_slices(images_tf)
	#-> write dataset to tfrecords file
	write_image_dataset_no_display(demo_dataset)
	#->read and display dataset samples
	display_image_tfrecords_dataset('demo_image_dataset.tfrecords')
