# a set of helpers to convert tensors to their serialized binary strings to facilitate message io accross processes
# from https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en
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

# Create a dictionary with features that may be relevant.
def serialize_image_float_example(image_tensor, label_id=None):
  
  feature = {
      'image_raw': _tensor_feature(image_tensor)#simpler and keeps dimensions info compared to _float_list_feature(image_tensor.numpy().flatten().tolist()),
  }
  if label_id is not None:
    feature.update({'label_id': _int64_feature_scalar(label_id)})

  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def serialize_tensor_label_example(tensor, label):
  
  feature = {
      'value': _tensor_feature(tensor),
      'label': _tensor_feature(label),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def serialize_float_with_text_label(values, label):
  """
  Creates a tf.train.Example string message ready to be written to a file or sent as a message.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'values': _float_feature(values),
      'label': _bytes_feature(label),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

###### Simple tensor + string label encode/decode helpers
def serialize_tensor_with_label(tensor, label):

  feature = {
      'values': _tensor_feature(tensor),
      'label': _bytes_feature(label),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
   
def decode_tensor_with_label_example(example_proto, dtype=tf.float32):
  ''' decode a tf.train.Example, supposing its structure is known '''
  # Create a dictionary describing the expected features.
  data_feature_description = {
      'values': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.string),
  }

  features_dict = tf.io.parse_single_example(example_proto, data_feature_description)
  print('features_dict', features_dict)
  instance_label=features_dict['label']
  instance_tensor=tf.io.parse_tensor(features_dict['values'], dtype)

  return instance_label, instance_tensor

###### Simple tensor encode/decode helpers
def serialize_tensor(tensor):
  ''' simply serialize a single tensor as a tf.Example '''
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
  ''' decode a tf.train.Example, supposing its structure is a simple tensor '''
  # Create a dictionary describing the expected features.
  data_feature_description = {
      'values': tf.io.FixedLenFeature([], tf.string),
  }

  features_dict = tf.io.parse_single_example(example_proto, data_feature_description)
  #print('features_dict', features_dict)
  instance_tensor=tf.io.parse_tensor(features_dict['values'], dtype, name='parse_values')

  return instance_tensor


def decode_multitensor_proto(data_feature_descriptions, example_proto):
  ''' decode a tf.train.Example, supposing its structure is known '''
  #print('+++++++++++++data_feature_descriptions', data_feature_descriptions['feature_specs'])
  features_dict = tf.io.parse_single_example(example_proto, data_feature_descriptions['feature_specs'])
  #print('features_dict', features_dict)
  instance_tensor={key_name: tf.io.parse_tensor(features_dict[key_name], data_feature_descriptions['types'][key_name], name='parse_'+key_name) for key_name in data_feature_descriptions['types'].keys()}
  '''for i in instance_tensor.keys():
    tf.print(i, tf.shape(instance_tensor[i]))
  '''
  return instance_tensor

def get_label_data_features_from_dataset(dataset):
  '''
  construct a dictionnary of feature description to decode serialized tensors
  '''  
  def get_features_specs(dataset_col_specs):
    data_feature_descriptions={}
    feature_types={}
    for keyID in dataset_col_specs.keys():
      print('item', keyID, dataset_col_specs[keyID])
      #data_feature_description
      #name:tf.io.FixedLenFeature([], tf.string)

      # Create a dictionary describing the expected features.
      data_feature_descriptions[keyID] = tf.io.FixedLenFeature([], tf.string)
      feature_types[keyID]=dataset_col_specs[keyID].dtype

    return {'feature_specs':data_feature_descriptions, 'types':feature_types}
  dataset_label_data_element_spec=dataset.element_spec
  label_specs=get_features_specs(dataset_label_data_element_spec[0])
  data_specs=get_features_specs(dataset_label_data_element_spec[1])
  print('label_specs=', label_specs)
  print('data_specs=', data_specs)
  return label_specs, data_specs
  



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
  

