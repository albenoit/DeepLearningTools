# ========================================
# FileName: tensor_msg_io.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of helpers to convert tensors to their serialized binary strings to facilitate message io accross processes
# from https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en
# for DeepLearningTools.
# =========================================

import tensorflow as tf

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(values):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature_scalar(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(values):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _tensor_feature(tensor):
  #first serialize the tensor
  serialized_nonscalar = tf.io.serialize_tensor(tensor)
  #next integrate into a tf.train.Feature
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_nonscalar.numpy()]))

#-----------------------------------------------------------------
# Create a dictionary with features that may be relevant.
#-----------------------------------------------------------------

def serialize_image_float_example(image_tensor, label_id=None):
  """
  Creates a tf.train.Example string message ready to be written to a file or sent as a message.

  :param image_tensor: The image tensor to be serialized.
  :type image_tensor: tf.Tensor

  :param label_id: The label associated with the image (optional).
  :type label_id: str

  :return: A dictionary mapping the feature name to the tf.train.Example-compatible.
  :rtype: str
  """
  feature = {
      'image_raw': _tensor_feature(image_tensor)#simpler and keeps dimensions info compared to _float_list_feature(image_tensor.numpy().flatten().tolist()),
  }
  if label_id is not None:
    feature.update({'label_id': _int64_feature_scalar(label_id)})

  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def serialize_tensor_label_example(tensor, label):
  """
  Creates a tf.train.Example string message ready to be written to a file or sent as a message.

  :param tensor: The tensor to be serialized.
  :type tensor: tf.Tensor

  :param label: The label associated with the tensor.
  :type label: str

  :return: A dictionary mapping the feature name to the tf.train.Example-compatible.
  :rtype: str
  """
  feature = {
      'value': _tensor_feature(tensor),
      'label': _tensor_feature(label),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def serialize_float_with_text_label(values, label):
  """
  Creates a tf.train.Example string message ready to be written to a file or sent as a message.

  :param values: The list of floating-point values to be serialized.
  :type values: list[float]

  :param label: The text label associated with the values.
  :type label: str

  :return: A dictionary mapping the feature name to the tf.train.Example-compatible.
  :rtype: str
  """
  feature = {
      'values': _float_feature(values),
      'label': _bytes_feature(label),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

#-----------------------------------------------------------
# Simple tensor + string label encode/decode helpers
#-----------------------------------------------------------

def serialize_tensor_with_label(tensor, label):
  """
  Creates a tf.train.Example string message ready to be written to a file or sent as a message.

  :param tensor: Input tensor to be serialized.
  :type tensor: tf.Tensor

  :param label: Corresponding label associated with the tensor.
  :type label: str

  :return: Create a dictionary describing the expected features.
  """
  feature = {
      'values': _tensor_feature(tensor),
      'label': _bytes_feature(label),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
   
def decode_tensor_with_label_example(example_proto, dtype=tf.float32):
  """
  Decode a tf.train.Example, supposing its structure is known.

  :param example_proto: Input tensor to be serialized.
  :type example_proto: tf.Tensor

  :return: Create a dictionary describing the expected features.
  """
  data_feature_description = {
      'values': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.string),
  }

  features_dict = tf.io.parse_single_example(example_proto, data_feature_description)
  print('features_dict', features_dict)
  instance_label=features_dict['label']
  instance_tensor=tf.io.parse_tensor(features_dict['values'], dtype)

  return instance_label, instance_tensor

#------------------------------------------------------
# Simple tensor encode/decode helpers
#------------------------------------------------------

def serialize_tensor(tensor):
  """
  Simply serialize a single tensor as a tf.Example.

  :param tensor: A tensor to be serialized.
  :type tensor: tf.Tensor
  
  :return: Create a dictionary describing the expected features.

  """
  if isinstance(tensor, (list,tuple)):
    #tf.print('*** input tensor is list or tuple')
    feature={'values_'+str(keyID):_tensor_feature(item) for keyID, item in enumerate(tensor)}
  elif isinstance(tensor, dict):
    #tf.print('*** input tensor is dict')
    feature={item:_tensor_feature(tensor[item]) for item in tensor.keys()}
  else: #if tensor is a classical tensor, then, encode with defauult name 'values'
    #tf.print('*** direct tensor encoding')
    feature = {
      'values': _tensor_feature(tensor),
    }

  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def decode_tensor_proto(example_proto, dtype):
  """
  Decode a tf.train.Example, supposing its structure is a simple tensor.

  :param example_proto: A dataset to be serialized.
  :type example_proto: tf.Tensor

  :return: A dictionary describing the expected features.
  """
  data_feature_description = {
      'values': tf.io.FixedLenFeature([], tf.string),
  }

  features_dict = tf.io.parse_single_example(example_proto, data_feature_description)
  #print('features_dict', features_dict)
  instance_tensor=tf.io.parse_tensor(features_dict['values'], dtype, name='parse_values')

  return instance_tensor


def decode_multitensor_proto(data_feature_descriptions, example_proto):
  """
  Decode a tf.train.Example, supposing its structure is known.

  :param data_feature_descriptions: A dataset to be serialized.
  :type data_feature_descriptions: tf.Tensor
  
  :return: Create a dictionary describing the expected features.
  """
  #print('+++++++++++++data_feature_descriptions', data_feature_descriptions['feature_specs'])
  features_dict = tf.io.parse_single_example(example_proto, data_feature_descriptions['feature_specs'])
  #print('features_dict', features_dict)
  instance_tensor={key_name: tf.io.parse_tensor(features_dict[key_name], data_feature_descriptions['types'][key_name], name='parse_'+key_name) for key_name in data_feature_descriptions['types'].keys()}
  '''for i in instance_tensor.keys():
    tf.print(i, tf.shape(instance_tensor[i]))
  '''
  return instance_tensor

def get_data_label_features_from_dataset(dataset):
  """
  Construct a dictionnary of feature description to decode serialized tensors.

  :param dataset: A dataset to be serialized.
  :type dataset: tf.Tensor
  
  :return: Create a dictionary describing the expected features.
  """
  def get_features_specs(dataset_col_specs, default_name):
    data_feature_descriptions={}
    feature_types={}
    if isinstance(dataset_label_data_element_spec[0], tf.TensorSpec):
      data_feature_descriptions[default_name] = tf.io.FixedLenFeature([], tf.string)
      feature_types[default_name]=dataset_col_specs.dtype
    else:

      for keyID in dataset_col_specs.keys():
        print('item', keyID, dataset_col_specs[keyID])
        #data_feature_description
        #name:tf.io.FixedLenFeature([], tf.string)

        # Create a dictionary describing the expected features.
        data_feature_descriptions[keyID] = tf.io.FixedLenFeature([], tf.string)
        feature_types[keyID]=dataset_col_specs[keyID].dtype

    return {'feature_specs':data_feature_descriptions, 'types':feature_types}
  dataset_label_data_element_spec=dataset.element_spec
  print('Dataset Feature specs (data, labels)', dataset_label_data_element_spec)
  data_specs=get_features_specs(dataset_label_data_element_spec[0], 'values')
  label_specs=get_features_specs(dataset_label_data_element_spec[1], 'label')
  print('label_specs=', label_specs)
  print('data_specs=', data_specs)
  return data_specs, label_specs
  
############### EXAMPLES ###################
#encoding example 1 (float scalar, label)
if __name__ == "__main__":
  serialized_example_1 = serialize_float_with_text_label(0.1, b'goat')
  print('serialized example_1', serialized_example_1)
  #related decoding example
  read_proto_1 = tf.train.Example.FromString(serialized_example_1)
  print('unserialized example_1', read_proto_1)

  #encoding example 2 (tensor, label)
  
  serialized_example_2 = serialize_tensor_with_label(tf.constant([0.1, 2.03], dtype=tf.float16), b'goat')
  print('serialized example_2', serialized_example_2)
  #related decoding example
  example_proto=tf.constant(serialized_example_2)
  #example = tf.train.Example()
  #example.ParseFromString(serialized_example_2)
  print('unserialized example_2', example_proto)
  decoded_example=decode_tensor_with_label_example(example_proto, tf.float16)
  print('unserialized example_2', decoded_example)
  

