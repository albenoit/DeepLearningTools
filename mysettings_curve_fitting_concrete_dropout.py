'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections #used to stack MC samples in the serving/test phase
import seaborn as sns
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
session_name='concrete_dropout'

''' define here some hyperparameters to adjust the experiment
===> Note that this dictionnary will complete the session name
'''
hparams={'hiddenNeurons':10,#set the number of neurons per hidden layers
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
XLA_FLAG=tf.OptimizerOptions.OFF#ON_1#OFF

#-> define here the used model under variable 'model'
#model_file='model_densenet.py'
model_file='model_curve_fitting_concrete_dropout.py'
field_of_view=20#unused
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
weights_weight_decay=0.0001
initial_learning_rate=0.1
num_epochs_per_decay=150 #number of epoch keepng the same learning rate
learning_rate_decay_factor=0.1 #factor applied to current learning rate when NUM_EPOCHS_PER_DECAY is reached
predict_using_smoothed_parameters=False#set True to use trained parameters values smoothed (EMA) along the training steps (better results expected BUT STILL DOES NOT WORK WELL IN THIS CODE VERSION)

#set here paths to your data used for train, val
#-> a first set of data
raw_data_dir_train = None
raw_data_dir_val = None
raw_data_filename_extension=None
nb_train_samples=1000 #manually adjust here the number of temporal items out of the temporal block size
nb_test_samples=1000
batch_size=200
reference_labels=['values']

test_repetitions=20
def numpycurve(x):
    sigma=1.0
    noise=np.random.normal(loc=0.0, scale=0.2, size=x.shape).astype(np.float32)
    x_neg=np.where(x<=0)
    x_pos=np.where(x>0)
    y=x.copy()
    y[x_neg]=x[x_neg]**2
    y[x_pos]=np.sqrt(x[x_pos])*5
    '''
    y=x**2
    '''
    return y+noise

def target_curve(x):
    ''' the function y=f(x) to learn
    Args:
       x: input values in the form of numpy array or tensorflow Tensors
    Return:
       y=f(x)
    '''
    #add noise and adapt to the context (Numpy or Tensorflow)
    #print('x='+str(x))
    if isinstance(x,tf.Tensor):
        #explicitely reshaping output to help graph construction
        y=tf.reshape(tf.py_func(numpycurve, [x], tf.float32), x.shape)
        return y

    elif isinstance(x,np.ndarray):
        return numpycurve(x)

    raise ValueError('Unsupported data type')

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
    # no preprocessing
    return features

def model_outputs_postprocessing_for_serving(model_outputs_dict):
    ''' define here the post-processings to be applied to each of the model outputs when used withtensorflow serving
        WARNING, in case of multiple outputs, ONE of them must be named as the
        default serving output: tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    Args:
        model_outputs_dict: the original model outputs dictionary
    Returns:
       the postprocessed outputs dictionnary
    '''
    #in this use case, we have two outputs:
    #->  code that is kept as is
    #->  semantic map logits from which we extract the most probable class index for each pixel
    postprocessed_outputs={model_head_embedding_name:model_outputs_dict['code'],
                           model_head_prediction_name:tf.concat([model_outputs_dict['prediction'],model_outputs_dict['log_var']],axis=1),
                           }
    return postprocessed_outputs

def getOptimizer(loss, learning_rate, global_step):
    '''define here the specific optimizer to be used
    '''
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

def get_total_loss(inputs, model_outputs_dict, labels, weights_loss):
    '''a specific loss for data reconstruction when dealing with autoencoders
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
        weights_loss: the model weights loss that may be used for regularization
    '''
    # computing the heteroscedastic loss
    precision = tf.exp(-model_outputs_dict['log_var'])
    """regularization losses:
    #reminder : on the dropout regularization side, one wants to MAXIMIZE the entropy of the Bernoulli random variable with probability 1-p
    ==> then this loss will be negative
    """
    reg_losses = tf.reduce_sum(tf.losses.get_regularization_losses())
    loss= tf.reduce_sum(precision * tf.square(labels - model_outputs_dict['prediction']) + model_outputs_dict['log_var']) + reg_losses#, -1)
    print('loss='+str(loss))
    tf.summary.scalar('ELBO', -0.5*loss)
    tf.summary.scalar('precision', tf.reduce_mean(precision))
    tf.summary.scalar('reg_losses', reg_losses)

    '''https://arxiv.org/pdf/1601.00670.pdf :
    The ELBO is the negative KLdivergence plus log(p(x), which is a constant with respect to q(z).
    Maximizing the ELBO is equivalent to minimizing the KL divergence.
    Examining the ELBO gives intuitions about the optimal variational density.'''
    return loss#+weights_weight_decay*weights_loss

def get_eval_metric_ops(inputs, model_outputs_dict, labels):
    '''Return a dict of the evaluation Ops.
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply with the ones defined in the model_file
        labels: the reference data / ground truth if available
    Returns:
        Dict of metric results keyed by name.
    '''
    return {
            'MSE': tf.metrics.mean_squared_error(
                labels=labels,
                predictions=model_outputs_dict['prediction'],
                name='mean_squared_error'),
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
        with tf.name_scope("generate_data"):
            # a simple uniform distribution centered on zero
            sampled_x = tf.random_uniform(shape=[batch_size,1], minval=-2, maxval=2)
            sampled_y=target_curve(sampled_x)
            print('input sample='+str(sampled_y))
        return sampled_x, sampled_y
    return input_fn, None

'''
################################################################################
## Serving (production) section, define here :
-get_input_pipeline_serving():  the input placeholder of the server that will receive the data
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
'''
def get_input_pipeline_serving():
    '''Build the serving inputs, expecting messages made of :
    -> a batch of size 1.
    -> a data buffer of type float32 of the same shape as each of the elements used along training (no preliminary normalisation is expected)
    '''
    serialized_tf_example = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, 1],
        name='serialized_input_data')

    return tf.estimator.export.ServingInputReceiver(
        serialized_tf_example, {input_data_name: serialized_tf_example})

def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max
D=1
def test(Y_true, MC_samples):
    """
    Estimate predictive log likelihood:
    log p(y|x, D) = log int p(y|x, w) p(w|D) dw
                 ~= log int p(y|x, w) q(w) dw
                 ~= log 1/K sum p(y|x, w_k) with w_k sim q(w)
                  = LogSumExp log p(y|x, w_k) - log K
    :Y_true: a 2D array of size N x dim
    :MC_samples: a 3D array of size samples K x N x 2*D with:
    --K = nb of test_repetitions
    --N = nb of samples sample(batch size ?)
    --D, the dimension of the ground truth data
    """
    assert len(MC_samples.shape) == 3
    assert len(Y_true.shape) == 2
    k = MC_samples.shape[0]
    N = Y_true.shape[0]
    D = MC_samples.shape[2]/2
    mean = MC_samples[:, :, D:]  # K x N x D
    logvar = MC_samples[:, :, :D]
    test_ll = -0.5 * np.exp(-logvar) * (mean - Y_true[None])**2. - 0.5 * logvar - 0.5 * np.log(2 * np.pi)
    test_ll = np.sum(np.sum(test_ll, -1), -1)
    test_ll = logsumexp(test_ll) - np.log(k)
    pppp = test_ll / N  # per point predictive probability
    rmse = np.mean((np.mean(mean, 0) - Y_true)**2.)**0.5
    return pppp, rmse

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
        self.repetition_count=0
        self.MC_samples= []#collections.deque(maxlen=test_repetitions)
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
        #generate a first set of data
        self.MC_samples= []
        self.genData()

    def genData(self):
        self.x=np.sort(np.random.uniform(low=-3, high=3, size=[batch_size,1]).astype(np.float32), axis=0)
        self.target=target_curve(self.x)
        if self.debugMode is True:
            print('Generated a new set of samples of shape '+str(self.target.shape))

    def getInputData(self, idx):
        ''' method that returns data samples complying with the placeholder
        defined in function get_input_pipeline_serving
        Args:
           idx: the input data index
        Returns:
           the data sample with shape and type complying with the server input
        '''
        #generate a new bunch of data sample each time the number of repetitions is reached
        if idx%test_repetitions==0:
            self.genData()
            self.MC_samples= []#reset MC samples list
        if self.debugMode is True:
            print('#sent new data, index='+str(idx))
        return self.x

    def plot(self, Y_val, means, mean_MC, epistemic_uncertainties, aleatoric_uncertainties, Y_target):
        X_val=self.x[:, 0]
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        for mean in means:
            self.ax1.scatter(X_val, mean[:, 0], c='b', alpha=0.2, lw=0)
        self.ax1.scatter(X_val, Y_target, c='r', lw=1, label='Noisy target')
        self.ax1.set_title('Target and MC samples')
        self.ax1.legend()
        #self.ax1.plot(X_val[indx, 0], np.mean(means, 0)[indx, 0], color='skyblue', lw=1)
        #sns.tsplot(data=means, time=X_val, ci=[25, 50, 75, 90], color="m", ax=self.ax2, n_boot=means.shape[0])
        for i in range(1,3):#sigma=[1,2,3]
            ymin, ymax = np.min(mean_MC - i * epistemic_uncertainties, axis=1), np.max(mean_MC + i * epistemic_uncertainties, axis=1)
            self.ax2.fill_between(X_val, ymin,ymax,color='skyblue',alpha=(4.-i)/4., label='std*'+str(i))#i/3)
        print('Xval.shape='+str(X_val.shape))
        print('ymin.shape='+str(ymin.shape))
        self.ax2.scatter(X_val, Y_target, c='r', lw=1, label='Noisy target')
        self.ax2.scatter(X_val, mean_MC, c='g', alpha=0.2, lw=0, label='MC average')
        self.ax2.legend()#self.ax2.scatter(X_val, ymin, c='b', alpha=0.2, lw=0)
        #self.ax2.scatter(X_val, ymax, c='b', alpha=0.2, lw=0)
        self.ax2.set_title('Prediction mean and epistemic uncertainty levels ')
        ymin_a, ymax_a = np.min(mean_MC - aleatoric_uncertainties, axis=1), np.max(mean_MC + aleatoric_uncertainties, axis=1)
        self.ax3.fill_between(X_val, ymin_a,ymax_a,color='skyblue',alpha=0.5, label='epistemic uncertainty')
        self.ax3.scatter(X_val, Y_target, c='r', lw=1, label='Noisy target')
        self.ax3.scatter(X_val, mean_MC, c='g', alpha=0.2, lw=0, label='MC mean')
        self.ax3.legend()#self.ax2.scatter(X_val, ymin, c='b', alpha=0.2, lw=0)
        self.ax3.set_title('Prediction mean and aleatoric uncertainty levels ')

        plt.pause(0.1)

    def decodeResponse(self, result):
        ''' receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        '''
        response = np.array(result.outputs[served_head].float_val)
        response=np.reshape(response, [batch_size,2])
        if self.debugMode is True:
            print('request shape='+str(self.x.shape))
            print('Answer shape='+str(response.shape))

        #process uncertainty measures when enough MC samples are collected
        self.MC_samples.append(response)
        if len(self.MC_samples)==test_repetitions:
            MC_samples = (np.array(self.MC_samples))
            print('---> PRETEST : (target, MCsamples) shapes='+str((self.target.shape, MC_samples.shape)))
            #---> PRETEST : (target, MCsamples) shapes=((200, 1), (20, 200, 2))
            pppp, rmse = test(self.target, MC_samples) # per point predictive probability
            D=MC_samples.shape[2]/2
            print('pppp, rmse='+str((pppp, rmse)))
            means = MC_samples[:, :, :D]  # K x N
            print('MC_samples means (KxN?)='+str(MC_samples.shape))
            epistemic_uncertainty_per_point =np.var(means, 0)
            mean_MC=np.mean(means,0)
            epistemic_uncertainty = epistemic_uncertainty_per_point.mean(0)
            logvar_per_point = np.exp(np.mean(MC_samples[:, :, D:], 0)) # per sample average logvar
            aleatoric_uncertainty = logvar_per_point.mean(0)
            print('Uncertainties: epistemic={epis} ; aleatoric={alea}'.format(epis=epistemic_uncertainty, alea=aleatoric_uncertainty))
            ps = np.array([sess.run(layer_p) for layer_p in tf.get_collection('LAYER_P')])
            self.plot(Y_val=response, means=means, mean_MC=mean_MC, epistemic_uncertainties=epistemic_uncertainty_per_point, aleatoric_uncertainties=logvar_per_point, Y_target=self.target)

    def finalize(self):
        ''' a function called when the prediction loop ends '''
        print('Prediction process ended successfuly')
