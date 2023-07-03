# ========================================
# FileName: model_serving_tools.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of helper functions to interract with a model server as a client
# for DeepLearningTools.
# =========================================

import os
import time
import configparser
import numpy as np
try:
  import tensorflow as tf
  from tensorflow_serving.apis import predict_pb2 #for single head models
  from tensorflow_serving.apis import inference_pb2 #for multi head models
  from tensorflow_serving.apis import prediction_service_pb2_grpc
  import grpc
  from grpc.framework.interfaces.face import face
  from tensorflow_serving.apis import prediction_service_pb2
except:
  print('Warning, tensorflow could not be loaded, some model_servin_tools may not work properly')

def decode_model_serving_answer(answer, output_names:list):
  """
  Classical decoding approach but still slow (numpy array creation from iterable is slow).

  :param answer: The model_server predict request response.
  :param output_names: The list of output names to be decoded.
  :return: The list of numpy arrays.
  """
  return [ tf.make_ndarray(answer.outputs[output]) for output in output_names]
    
def get_model_server_cfg(model_dir):
  """
  Read model server configuration from the model_serving_setup.ini written in the target experiment folder.

  :param model_dir: Path to an experiment (trained model).
  :return: A dictionary that describes the expected server configuration.
  """
  model_server_config=model_dir+'/model_serving_setup.ini'
  if os.path.exists(model_server_config) == False:
    raise ValueError("Config file does not exist")
  config=configparser.ConfigParser()
  config.read(model_server_config, encoding='utf8')
  print(config.keys())
  print('trained model serving config:', config)
  return config

def get_served_model_info(one_model_path, expected_model_name):
  """
  Basic function that checks served model behaviors.

  :param one_model_path: The path to a servable model directory.
  :param expected_model_name: The model name that is expected to be found on the server.
  :return: Nothing for now.
  """
  import subprocess
  #get the first subfolder of the served models directory
  served_model_info_cmd='saved_model_cli show --dir {target_model} --tag_set serve --signature_def {model_name}'.format(target_model=one_model_path,
                                                                                      model_name=expected_model_name)
  print('Checking served model available signatures using command '+served_model_info_cmd)
  cmd_result=subprocess.check_output(served_model_info_cmd.split())
  print('Answer:')
  print(cmd_result.decode())
  if expected_model_name in cmd_result.decode():
    print('Target model {target} name found in the command answer'.format(target=expected_model_name))
  else:
    raise ValueError('Target model {target} name NOT found in the command answer'.format(target=expected_model_name))

def setup_model_server_connexion(host, port, grpc_max_message_length=0):
  """
  Set up the connection to the model server for prediction.

  Test scripts may help : https://github.com/tensorflow/serving/blob/master/tensorflow_serving/model_servers/tensorflow_model_server_test.py

  :param host: The host address of the model server.
  :param port: The port number of the model server.
  :param grpc_max_message_length: Maximum message length for gRPC communication.
  :return: The prediction service stub for making predictions.
  """
  print('Trying to interract with server:{srv} on port {port} for prediction...'.format(srv=host,
                                                        port=port))
  server=host+':'+str(port)
  # specify option to support messages larger than allowed by default
  grpc_options=None
  if grpc_max_message_length !=0:
      grpc_options = [('grpc.max_send_message_length', grpc_max_message_length)]
      grpc_options = [('grpc.max_receive_message_length', grpc_max_message_length)]
  channel = grpc.insecure_channel(server, options=grpc_options)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  return stub

def generate_single_request(sample:dict, model_name:str, debug:bool=False):
  """
  Build and send a single request to the server.

  :param sample: A dictionary with keys as input tensor names and values as numpy arrays.
  :param model_name: The name of the model.
  :param debug: Whether to print debug information.
  :return: The predict request.
  """
  if not(isinstance(sample, dict)):
        raise ValueError('Expecting a dictionnary of values that will further be converted to proto buffers. Dictionnary keys must correspond to the usersettings.served_input_names strings list')
  if debug:
    start_time=time.time()

  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = model_name#experiment_settings.served_head_names[0]
  for inputname in sample.keys():
      feature=sample[inputname]
      feature_proto=tf.make_tensor_proto(feature, shape=feature.shape)
      request.inputs[inputname].CopyFrom(feature_proto)
  if debug:
      print('Time to prepare request:',round(time.time() - start_time, 2))
  return request

def WaitForServerReady(usersettings, host, port):
  """
  Waits for a server on the localhost to become ready.

  Reference: inspired from https://github.com/tensorflow/serving/blob/master/tensorflow_serving/model_servers/tensorflow_model_server_test.py

  :param usersettings: The user settings.
  :param host: The TensorFlow server address.
  :param port: The port address of the PredictionService.
  :return: True if the server is ready, False on timeout.
  """
  #FIXME fix the following imports that may be deprecated:
  from grpc import implementations

  for _ in range(0, usersettings.wait_for_server_ready_int_secs):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'server_not_real_model_name'

    try:
      # Send empty request to missing model
      print('Trying to reach tensorflow-server {srv} on port {port} for {delay} seconds'.format(srv=host,
                                                             port=port,
                                                             delay=usersettings.wait_for_server_ready_int_secs))
      channel = implementations.insecure_channel(host, int(port))
      stub = prediction_service_pb2.PredictionServiceStub(channel)
      stub.Predict(request, 1)
    except face.AbortionError as error:
      # Missing model error will have details containing 'Servable'
      if 'Servable' in error.details:
        print('Server is ready')
        return True
      else:
        print('Error:'+str(error.details))
    return False
    time.sleep(1)

def _create_rpc_callback(client, debug):
  """
  Creates an RPC callback function.

  :param client: A ClientIO instance.
  :param debug: A boolean indicating whether to print debug information.
  :return: The callback function.
  """
  def _callback(result_future):
    """
    Callback function.
    Calculates the statistics for the prediction result.

    :param result_future: Result future of the RPC.
    """
    print('Received response:'+str(result_future))
    exception = result_future.exception()
    if exception:
      #result_counter.inc_error()
      print(exception)
    else:
      try:
          if debug:
              print(result_future.result())
          client.decodeResponse(result_future.result())
      except Exception as e:
          raise ValueError('Exception encountered on client callback : '.format(error=e))
  return _callback

#-----------------------------------------------
# fast protobuf answer to numpy array
#-----------------------------------------------

#@tf.function(jit_compile=True, reduce_retracing=True)
def deserialize_srv_answer_uint8(proto_single_output):
  """
  Deserialize a protobuf output into a tensor.

  :param proto_single_output: A protobuf output supposed to be a serialized tensor of type tf.uint8.
  :return: The deserialized tensor.
  """
  shape = tf.TensorShape(proto_single_output.tensor_shape)
  output = tf.io.parse_tensor(tf.reshape(proto_single_output.string_val, shape), out_type=tf.uint8)
  return output

def deserialize_srv_answer_uint8_vfrombuffer(msg_buffer, shape):
  """
  Faster but less elegant method to deserialize a protobuf output into a tensor.

  :param msg_buffer: The protobuf message buffer.
  :param shape: The shape of the tensor.
  :return: The deserialized tensor.
  """
  out = np.frombuffer(bytearray(msg_buffer.string_val[0]),dtype=np.uint8,count=-1,offset=27).reshape(shape)
  return out
