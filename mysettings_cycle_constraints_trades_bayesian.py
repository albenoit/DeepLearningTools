'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
TODO, check with : https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb
https://machinelearningmastery.com/remove-trends-seasonality-difference-transform-python/
'''
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataProvider_input_pipeline
import model_utils
#-> set here your own working folder
workingFolder='experiments/curves_fitting'

#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=1#timeout limit when a client requests a served model

'''if save_model_variables_to_pandas=True, then force to save all model variables to a pandas dataframe file named 'model_parameters.bz2'
To load them later, do (update the path to your experiment):
import pandas
a=pandas.read_pickle('experiments/curves_fitting/my_test_2018-02-12--17:48:17/model_parameters.bz2')
'''
save_model_variables_to_pandas=True

#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='LSTM'

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
XLA_FLAG=tf.OptimizerOptions.OFF#ON_1#OFF

#-> define here the used model under variable 'model'
#model_file='model_densenet.py'
model_file='model_cycle_constraints_trades_LSTM.py'#TCN_bayesian.py'
field_of_view=0#29
display_model_layers_info=False
#-> define here a string name used for the train, eval and served models
input_data_name='input'
model_head_embedding_name='prediction'
model_head_prediction_name=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY#'prediction'
#->define here the output that will be provided by tensorflow-server
served_head=model_head_prediction_name

#-> set the number of summaries store per training epoch (more=more precise BUT higer cost)
nb_summary_per_train_epoch=4

#define image patches extraction parameters
patchSize=224

#random seed used to init weights, etc. Use an integer value to make experiments reproducible
random_seed=42

# learning rate decaying parameters
nbEpoch=300
weights_weight_decay=0.01
initial_learning_rate=0.0001
num_epochs_per_decay=150 #number of epoch keepng the same learning rate
learning_rate_decay_factor=0.1 #factor applied to current learning rate when NUM_EPOCHS_PER_DECAY is reached
grad_clip_norm=1.0
predict_using_smoothed_parameters=False#set True to use trained parameters values smoothed (EMA) along the training steps (better results expected BUT STILL DOES NOT WORK WELL IN THIS CODE VERSION)

#set here paths to your data used for train, val
#-> a first set of data
raw_data_dir_train = '/home/alben/workspace/Datasets/Trading/train'
raw_data_dir_val = '/home/alben/workspace/Datasets/Trading/val'
raw_data_filename_extension='*.txt'
csv_field_delim='\t'
nb_train_samples=117234 #manually adjust here the number of temporal items out of the temporal block size
nb_test_samples=10000
batch_size=10
MC_repeats=40
nb_classes=2
time_series_length=29#field_of_view
record_defaults=[["timestamp"], [0.0]]
reference_labels=['startDate', 'stopDate'] #to be used if many labels are generated by the get_input_pipeline_train_val function

def get_ROI(tensor, direction, length=time_series_length):
    if direction == 'forward':# forward time prediction : avoiding the first time steps covered by the field of view
        return tf.slice(tensor, begin=[0,field_of_view,0], size=[-1,length, -1])
    elif  direction == 'backward':#backward(reverse) time prediction : avoiding the last time steps covered by the field of view
        return tf.slice(tensor, begin=[0,0,0], size=[-1,length, -1])
    else:
        raise ValueError('Expected direction parameter string \'forward\' OR \'backward\' ')

####################################################
## Define here use case specific metrics, loss, etc.
#with tf.name_scope("loss"):
def data_preprocess(features, model_placement):
    ''' define here the chosen data preprocessing that will be applied
    all the time, for training, validation and serving
    Manually specify here on which device this preprocessing should be done.
    For convenience, the placement of the model that follows this step is also provided
    so that you may want to place it on the same device.
    Args:
        features: the input data that is being processed
        model_placement: the device where the following model will be placed
    Returns:
       the preprocessed data
    '''
    #reshape data to a 4d shape to apply classical 2D ops(and the row dimension will be 1)
    #data=tf.expand_dims(features, 1)
    return features#data

model_serving_output_vectors_length=time_series_length+field_of_view*2
def model_outputs_postprocessing_for_serving(model_outputs_dict):
    ''' define here the post-processings to be applied to each of the model outputs when used withtensorflow serving
        WARNING, in case of multiple outputs, ONE of them must be named as the
        default serving output: tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    Args:
        model_outputs_dict: the original model outputs dictionary
    Returns:
       the postprocessed outputs dictionnary
    '''
    #in this use case, we have four concatenated outputs:
    outputs_concat=tf.concat([  get_ROI(model_outputs_dict['F_x'], 'backward', length=model_serving_output_vectors_length),
                                get_ROI(model_outputs_dict['G_y'], 'backward', length=model_serving_output_vectors_length),
                                get_ROI(model_outputs_dict['GoF_x'], 'backward', length=model_serving_output_vectors_length),
                                get_ROI(model_outputs_dict['FoG_y'], 'backward', length=model_serving_output_vectors_length)], axis=1)
    postprocessed_outputs={model_head_embedding_name:model_outputs_dict['F_x'],
                           model_head_prediction_name:outputs_concat,
                           }
    return postprocessed_outputs

def getOptimizer(loss, learning_rate, global_step):
    '''define here the specific optimizer to be used
    '''
    #get gradient summary information and the gradient norm
    tvars, raw_grads, gradient_norm=model_utils.track_gradients(loss)

    #clip them wrt the max allowed norm
    if grad_clip_norm>0:
      grads, _ = tf.clip_by_global_norm(raw_grads, clip_norm=grad_clip_norm, use_norm=gradient_norm)
    else:
      grads=raw_grads

    #setup solver
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    #update weights with the clipped gradient
    optimizer_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    return optimizer_op
    #return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    #MomentumOptimizer, AdamOptimizer, GradientDescentOptimizer

    #gradient clipping:https://stackoverflow.com/questions/35226428/how-do-i-get-the-gradient-of-the-loss-at-a-tensorflow-variable
    #https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow
    #  https://www.tensorflow.org/versions/master/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
    #https://machinelearningmastery.com/exploding-gradients-in-neural-networks/

def get_total_loss(inputs, model_outputs_dict, labels, weights_loss):
    '''a specific loss for data reconstruction when dealing with autoencoders
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
        weights_loss: the model weights loss that may be used for regularization
    '''
    #selecting in the first half of the sequence, the ones AFTER model field of view
    X=tf.slice(inputs, begin=[0,field_of_view,0], size=[-1,time_series_length, -1], name='first_period')
    #selecting in the second half of the sequence, the ones BEFORE model field of view
    Y=tf.slice(inputs, begin=[0,time_series_length+field_of_view,0], size=[-1,time_series_length, -1], name='second_period')
    #Y=tf.Print(Y, [tf.layers.flatten(inputs)], message="Inputs=", summarize=2*time_series_length+2*field_of_view)
    #Y=tf.Print(Y, [tf.layers.flatten(X)], message="X=", summarize=field_of_view+time_series_length)
    #Y=tf.Print(Y, [tf.layers.flatten(Y)], message="Y=", summarize=time_series_length+field_of_view)
    #
    #Y=tf.Print(Y, [tf.layers.flatten(X)], message="X=", summarize=2*time_series_length)
    #Y=tf.Print(Y, [tf.layers.flatten(Y)], message="Y=", summarize=2*time_series_length)
    #Y=tf.Print(Y, [tf.layers.flatten(model_outputs_dict['F_x'])], message="Y_est_all=", summarize=time_series_length+field_of_view)
    #Y=tf.Print(Y, [tf.layers.flatten(get_ROI(model_outputs_dict['F_x'], 'forward'))], message="Y_est_roi=", summarize=time_series_length)
    #Y=tf.Print(Y, [model_outputs_dict['G_y'], get_ROI(model_outputs_dict['G_y'], 'backward')], message="X_all,X_backward", summarize=2*time_series_length)
    #print('get_total_loss : X,Y slices ='+str((X,Y)))

    loss_F=tf.losses.mean_squared_error(
                            labels=Y,
                            predictions=get_ROI(model_outputs_dict['F_x'], 'forward'),
                            scope='forward_prediction_error')
    loss_G=tf.losses.mean_squared_error(
                            labels=X,
                            predictions=get_ROI(model_outputs_dict['G_y'], 'backward'),
                            scope='backward_prediction_error')
    loss_F_G=tf.losses.mean_squared_error(
                            labels=Y,
                            predictions=get_ROI(model_outputs_dict['FoG_y'], 'backward'),
                            scope='backward_forward_prediction_error')
    loss_G_F=tf.losses.mean_squared_error(
                            labels=X,
                            predictions=get_ROI(model_outputs_dict['GoF_x'], 'forward'),
                            scope='forward_backward_prediction_error')

    #loss_F=tf.reduce_sum(tf.square(Y - get_ROI(model_outputs_dict['F_x'], 'forward')))
    #print('loss_F='+str(loss_F))
    totalLoss=loss_F+loss_G+loss_F_G+loss_G_F
    #totalLoss=tf.Print(loss_F,[loss_F], 'loss')
    tf.summary.scalar('task_weights_loss_ratio', weights_loss/loss_F)
    return totalLoss+weights_weight_decay*weights_loss

def get_eval_metric_ops(inputs, model_outputs_dict, labels):
    '''Return a dict of the evaluation Ops.
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
    Returns:
        Dict of metric results keyed by name.
    '''

    #selecting in the first half of the sequence, the ones AFTER model field of view
    X=tf.slice(inputs, begin=[0,field_of_view,0], size=[-1,time_series_length, -1], name='first_period')
    #selecting in the second half of the sequence, the ones BEFORE model field of view
    Y=tf.slice(inputs, begin=[0,time_series_length+field_of_view,0], size=[-1,time_series_length, -1], name='second_period')

    return {
            'MSE_F': tf.metrics.mean_squared_error(
                labels=Y,
                predictions=get_ROI(model_outputs_dict['F_x'], 'forward'),
                name='mean_squared_error_F'),
            'MSE_G': tf.metrics.mean_squared_error(
                labels=X,
                predictions=get_ROI(model_outputs_dict['G_y'], 'backward'),
                name='mean_squared_error_G'),
            'MSE_FoG': tf.metrics.mean_squared_error(
                labels=Y,
                predictions=get_ROI(model_outputs_dict['FoG_y'], 'backward'),
                name='mean_squared_error_FoG'),
            'MSE_GoF': tf.metrics.mean_squared_error(
                labels=X,
                predictions=get_ROI(model_outputs_dict['GoF_x'], 'forward'),
                name='mean_squared_error_GoF'),
            'MSE_GoF_FoG': tf.metrics.mean_squared_error(
                labels=get_ROI(model_outputs_dict['FoG_y'], 'backward'),
                predictions=get_ROI(model_outputs_dict['GoF_x'], 'forward'),
                name='mean_squared_error_GoF'),
            }

'''Define here the input pipelines :
-1. a common function for train and validation modes
-2. a specific one for the serving model_extra_update_ops
'''
def get_input_pipeline_train_val(batch_size, raw_data_files_folder, shuffle_batches):
    ''' define an input pipeline able to load temporal series from a set of
    CSV files and a batch size specified as inputs
    TODO, look at the doc here : https://www.tensorflow.org/programmers_guide/datasets
    @param batch_size : the expected size of a batch
    @param raw_data_files_folder : the folder where CSV files are stored
    @param shuffle_batches : a boolean that activates batch shuffling
    '''
    def input_fn():
        #load all csv files to use for training
        raw_data_files=DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_files_folder, file_extension=raw_data_filename_extension)
        print('Input files found='+str(raw_data_files))

        with tf.name_scope("retrieve_data"):
            """#FAKE DATA
            with tf.name_scope("generate_data"):
                # a simple uniform distribution centered on zero
                start_point = tf.random_uniform(shape=[batch_size,1], minval=-3*time_series_length, maxval=time_series_length)
                x_points=start_point+tf.lin_space(start=0.0, stop=time_series_length*2, num=time_series_length*2)
                # function to model is specified here:
                #curve=tf.sin(start_point+linear_data_x)
                curve=target_curve(x_points)
                curve=tf.Print(curve,[x_points, curve], message='x,f(x)=')
                print('input sample='+str(curve))
            return curve, start_point
            """
            #REAL DATA FROM FILE
            data_provider, iterator_initializer_hook=DataProvider_input_pipeline.FileListProcessor_csv_time_series(files=raw_data_files,
                                                                                 csv_field_delim=csv_field_delim,
                                                                                 record_defaults_values=record_defaults,
                                                                                 nblines_per_block=time_series_length*2+2*field_of_view,
                                                                                 queue_capacity=batch_size*5,
                                                                                 shuffle_batches=False)
            timestamps, single_period_data_block_raw=data_provider.dequeue_many(batch_size)
            '''
            one label per sample example:
            timestamps_start_stop=tf.string_join([timestamps[:,1],timestamps[:,-1]], separator='->')
            '''
            '''
            two labels per sample example:
            '''
            timestamps_start_stop=tf.stack([timestamps[:,1],timestamps[:,-1]],1)
            #raw_input('timestamps_start_stop='+str(timestamps_start_stop))
        return single_period_data_block_raw, timestamps_start_stop
    return input_fn, None#iterator_initializer_hook

'''
################################################################################
## Serving (production) section, define here :
-get_input_pipeline_serving():  the input placeholder of the server that will receive the data
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
'''
time_series_input_serving_shape=[1, field_of_view*2+time_series_length,1]
def get_input_pipeline_serving():
    '''Build the serving inputs, expecting messages made of :
    -> a batch of size 1.
    -> a data buffer of type float32 of the same shape as each of the elements used along training (no preliminary normalisation is expected)
    '''
    serialized_sample = tf.placeholder(
        dtype=tf.float32,
        shape=time_series_input_serving_shape,
        name='serialized_input_data')

    #replicating the sample multiple times to perform MCMC
    tiled_sample=tf.tile(serialized_sample, multiples=[MC_repeats,1,1])

    return tf.estimator.export.ServingInputReceiver(
        tiled_sample, {input_data_name: serialized_sample})

class Client_IO:
    ''' A specific class dedicated to clients that need to interract with
    a Tensorflow server that runs the above model
    --> must have the following methods:
    def __init__(self, debugMode): constructor that receives a debug flag
    def getInputData(self, idx): that generates data to send to the server
    def decodeResponse(self, result): that receives the response
    '''
    def __init__(self, debugMode):
        ''' constructor
            Args:
               debugMode: set True if some debug messages should be displayed
        '''
        self.debugMode=debugMode
        if self.debugMode is True:
            print('RPC Client ready to interract with the server')

        #load the test csv file and stack into memory
        #self.inputdata=np.genfromtxt(os.path.join(raw_data_dir_val,'CAC_norm.txt'), delimiter=csv_field_delim)
        self.inputdata=pd.read_csv(os.path.join(raw_data_dir_val,'CAC_norm.txt'), delimiter=csv_field_delim).as_matrix()
        #print('Read text data, shape='+str(self.inputdata.shape))
        self.neighborhood_range=2
        self.current_time_idx=self.neighborhood_range*time_series_length+field_of_view
        #prepare plots
        self.fig, self.ax = plt.subplots()
        self.firstCall=True
        assert self.neighborhood_range>=2, "data neighborhood range used for plotting must be at least 2"

    def getInputData(self, idx):
        ''' method that returns data samples complying with the placeholder
        defined in function get_input_pipeline_serving
        Args:
           idx: the input data index
        Returns:
           the data sample with shape and type complying with the server input
        '''
        #get the x curve ticks and related data block
        t_start=self.current_time_idx-time_series_length*self.neighborhood_range
        t_stop=self.current_time_idx+time_series_length*(self.neighborhood_range+1)
        self.current_data_block=self.inputdata[t_start:t_stop,:]
        self.x=self.current_data_block[:,0]
        self.target=self.current_data_block[:,1]

        #focus on the request
        present_start=time_series_length*self.neighborhood_range #an offset from tt_start to reach the index of self.current_time_idx
        present_stop=present_start+time_series_length
        request_t_start=present_start-field_of_view
        request_t_stop=request_t_start+time_series_input_serving_shape[1]

        self.x_current=self.x[present_start:present_stop]
        self.x_previous=self.x[present_start-(time_series_length+field_of_view):present_start]
        self.x_next=self.x[present_stop:present_stop+(time_series_length+field_of_view)]
        #print('x.previous='+str(self.x_previous))
        #print('x.current='+str(self.x_current))
        #print('x.next='+str(self.x_next))
        #increment time steps
        self.current_time_idx+=1

        if self.debugMode is True:
            print('Generating input features (random values) of shape '+str(self.target.shape))
        #send data to estimation server
        self.request_xticks=np.reshape(self.x[request_t_start:request_t_stop], time_series_input_serving_shape).astype(np.float32)
        self.request_values=np.reshape(self.target[request_t_start:request_t_stop], time_series_input_serving_shape).astype(np.float32)
        return self.request_values

    def decodeResponse(self, result):
        ''' receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        '''
        response_raw = np.array(result.outputs[served_head].float_val)
        responses=np.reshape(response_raw, [MC_repeats, 4, model_serving_output_vectors_length])#+2*field_of_view])
        self.f_x=responses[:,0,field_of_view:]##,field_of_view:field_of_view+time_series_length]
        last_id=time_series_input_serving_shape[1]-field_of_view
        self.g_y=responses[:,1,:last_id]##,:time_series_length]#eq to : :-2*field_of_view]
        #print('self.g_y',self.g_y,self.g_y.shape)
        self.gof_x=responses[:,2,field_of_view:last_id]#,field_of_view:field_of_view+time_series_length]
        self.fog_y=responses[:,3,field_of_view:last_id]#,field_of_view:field_of_view+time_series_length]

        if self.debugMode is True:
            print('self.f_x shape='+str(self.f_x))
            print('Response.shape='+str(response_raw))
            print('BEFORE self.x[:time_series_length]='+str(self.x[:time_series_length]))
            print('NOW    self.x[time_series_length:2*time_series_length]='+str(self.x[time_series_length:2*time_series_length]))
            print('AFTER  self.x[2*time_series_length:]='+str(self.x[2*time_series_length:]))
            print('request shape='+str(self.target.shape))
            print('Answer shape='+str(responses.shape))

        def plot_MCsamples(ax, x_ticks, MCsamples, plot_color, plot_label):
          #uncertainties estimation
          #D=self.f_x.shape[2]/2
          #print('pppp, rmse='+str((pppp, rmse)))
          mc_sample_pred = MCsamples[:, :]#, :D]  # K x N
          #print('MC_samples means (KxN?)='+str(MC_samples.shape))
          epistemic_uncertainty_per_point =np.var(mc_sample_pred, 0)*20
          mean_MC=np.mean(mc_sample_pred,0)
          epistemic_uncertainty = epistemic_uncertainty_per_point.mean(0)
          #print('epristemic uncertainty ='+str(epistemic_uncertainty_per_point))#logvar_per_point = np.exp(np.mean(MC_samples[:, :, D:], 0)) # per sample average logvar
          #print('epristemic uncertainty shape='+str(epistemic_uncertainty_per_point.shape))#logvar_per_point = np.exp(np.mean(MC_samples[:, :, D:], 0)) # per sample average logvar
          #print("x_ticks.shape="+str(x_ticks.shape))
          #print("mc_sample_pred.shape="+str(mc_sample_pred.shape))
          #print("mean_MC.shape="+str(mean_MC.shape))
          #aleatoric_uncertainty = logvar_per_point.mean(0)
          #print('Uncertainties: epistemic={epis} ; aleatoric={alea}'.format(epis=epistemic_uncertainty, alea=aleatoric_uncertainty))
          #ps = np.array([sess.run(layer_p) for layer_p in tf.get_collection('LAYER_P')])
          for i in range(1,3):#sigma=[1,2,3]
              #ymin, ymax = np.min(mean_MC - i * epistemic_uncertainty_per_point, axis=1), np.max(mean_MC + i * epistemic_uncertainty_per_point, axis=1)
              ymin, ymax = mean_MC - i * epistemic_uncertainty_per_point, mean_MC + i * epistemic_uncertainty_per_point
              self.ax.fill_between(x_ticks, ymin,ymax,color='skyblue',alpha=(4.-i)/4.)#, label='std*'+str(i))#i/3)
          self.ax.plot(x_ticks, mean_MC, plot_color,label=plot_label+' MC mean')

        self.ax.cla()
        self.ax.plot(self.x, self.target,'r-',label='target')
        self.ax.plot(self.request_xticks.flatten(), self.request_values.flatten(),'k+',label='submitted request')
        #self.ax.plot(self.x_next, self.f_x.transpose(),'g--',label='F(x), predict next')
        if self.debugMode:
          print('self.x_next, self.f_x='+str((self.x_next.shape, self.f_x.shape)))
        plot_MCsamples(self.ax, self.x_next, self.f_x,'g--', 'F(x), predict next')
        plot_MCsamples(self.ax, self.x_previous, self.g_y, 'c--', 'G(y), predict past')
        #self.ax.plot(self.x_previous, self.g_y.transpose(),'c--',label='G(y), predict past')

        plot_MCsamples(self.ax, self.x_current, self.gof_x, 'm--', 'GoF(x), reconstruct')
        #self.ax.plot(self.x_current, self.gof_x.transpose(),'m--',label='GoF(x), reconstruct')
        plot_MCsamples(self.ax, self.x_current, self.fog_y, 'b--', 'FoG(y), reconstruct')
        #self.ax.plot(self.x_current, self.fog_y.transpose(),'b--',label='FoG(y), reconstruct')
        self.ax.legend()
        if self.firstCall is True:
            self.ax.legend()
            self.firstCall=False
        plt.pause(0.02)

    def finalize(self):
        ''' a function called when the prediction loop ends '''
        print('Prediction process ended successfuly')
