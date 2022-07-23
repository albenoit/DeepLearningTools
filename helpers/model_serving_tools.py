""" 
A set of helper functions to interract with a model server as a client
A. Benoit, LISTIC Lab, 2022
"""
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
    """ classical decoding approach but still slow (numpy array creation from iterable is slow 
    Args: 
     answer: the model_server predict request response
     output_names: the list of output names to be decodes
    Returns the list of numpy arrays
    """
    return [ tf.make_ndarray(answer.outputs[output]) for output in output_names]
    
def get_model_server_cfg(model_dir):
    """ read model server configuration from the model_serving_setup.ini written in the target experiment folder
        Args: model_dir: path to an experiment (trained model)
        Returns: a dictionnary that describes the expected server configuration 
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
  ''' basic function that checks served model behaviors
  Args:
  one_model_path: the path to a servable model directory
  expected_model_name: the model name that is expected to be found on the server
  Returns:
    Nothing for now
  '''
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

    print('Trying to interract with server:{srv} on port {port} for prediction...'.format(srv=host,
                                                         port=port))
    ''' test scripts may help : https://github.com/tensorflow/serving/blob/master/tensorflow_serving/model_servers/tensorflow_model_server_test.py
    '''

    server=host+':'+str(port)
    # specify option to support messages larger than alloed by default
    grpc_options=None
    if grpc_max_message_length !=0:
        grpc_options = [('grpc.max_send_message_length', grpc_max_message_length)]
        grpc_options = [('grpc.max_receive_message_length', grpc_max_message_length)]
    channel = grpc.insecure_channel(server, options=grpc_options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub

def generate_single_request(sample:dict, model_name:str, debug:bool=False):
    """ build and send a single request to the server
      Args:
        sample: a disctionnary with keys:values= input_tensor_name:numpy array
        usersettings: the experiment settings
        
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
  #inspired from https://github.com/tensorflow/serving/blob/master/tensorflow_serving/model_servers/tensorflow_model_server_test.py
  """Waits for a server on the localhost to become ready.
  returns True if server is ready or False on timeout
  Args:
      host:tensorfow server address
      port: port address of the PredictionService.
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
  """Creates RPC callback function.
  Args:
    client: a CLientIO instance
    debug: a boolean, if True prints some debug information
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.
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

