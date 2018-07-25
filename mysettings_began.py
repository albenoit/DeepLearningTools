'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
Here is an example of the BEGAN model https://arxiv.org/pdf/1703.10717.pdf
TODO: GANs are difficult to train, one should have a look at some recommendations such as:
https://github.com/soumith/ganhacks
https://mlnotebook.github.io/post/GAN4/
'''
import tensorflow as tf
import numpy as np

DEBUG_OPTIM=False#set True to activate some Prints on the monitored variables along training

#-> set here your own working folder
workingFolder='experiments/generative'

#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=1#timeout limit when a client requests a served model
#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='BEGAN_'

#-> allow X window displays (for image and graph display purpose)
allow_display=True

#-> activate session profiling to observe ressource use and timings
do_trace_computation=True

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
model_file='model_began.py'
display_model_layers_info=False #a flag to enable the display of additionnal console information on the model properties (for debug purpose)

field_of_view=28

test_patch_overlapping_ratio=0.75 #-> patch overlapping when evaluating/predicting

#-> define here a string name used for the train, eval and served models
input_data_name='input'
model_head_generator_name=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
model_head_prediction_name='energy'
served_head=model_head_generator_name #define here the output that will be provided by tensorflow-server

#-> define the training strategy depending on the computing architecture
#---> "continuous_train_and_eval"-> single machine
#---> "train_and_evaluate" -> multiple machines/distributed training/evaluation
train_val_schedule_strategy="continuous_train_and_eval"

#-> set the number of summaries store per training epoch (more=more precise BUT higer cost)
nb_summary_per_train_epoch=-1
summary_fake_samples_max_number=3

#define image patches extraction parameters
patchSize=28

#random seed used to init weights, etc. Use an integer value to make experiments reproducible
random_seed=None

# learning rate decaying parameters
nbEpoch=100
weights_weight_decay=0.0001
initial_learning_rate=1e-5
num_epochs_per_decay=100 #FIXME, initial value = 3000 #number of epoch keepng the same learning rate
learning_rate_decay_factor=0.95 #factor applied to current learning rate when NUM_EPOCHS_PER_DECAY is reached
predict_using_smoothed_parameters=False#set True to use trained parameters values smoothed along the training steps (better results expected BUT STILL DOES NOT WORK WELL IN THIS CODE VERSION)
#set here paths to your data used for train, val, testraw_data_dir_train = "/home/alben/workspace/Datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/"
#-> train and val sets of data
from tensorflow.examples.tutorials.mnist import input_data
mnist_loc='datasamples/mnist/'
raw_data_dir_train=mnist_loc
raw_data_dir_val=mnist_loc
mnist = input_data.read_data_sets(mnist_loc, one_hot=False)
nb_train_samples=64#mnist.train.num_examples #nb_train_images*number_of_crops_per_image# number of images * number of crops per image
nb_test_samples=64#mnist.test.num_examples#to be adjusted for testing
embedding_samples_stored_number=nb_test_samples
batch_size=32
nb_classes=64 #here, this will correspond to the generator code size
reference_labels=['digit_values']
''' BEGAN specific optimization parameters :'''
equilibrium_gamma=0.8
optimizer_beta1 = 0.5
lambd_k = 1e-3
lr_lower_bound = 2e-5

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
    # do nothing, train and val input pipeline standardize data on their own and
    # serving will do its own too
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
    #in this use case, we expand the 2D generator output dynamic to range [0;255]
    # and cast to uint8 to display images
    with tf.name_scope('convert_to_standard_image'):
        eps=0.001
        fake_min= tf.reduce_min(model_outputs_dict['generator_fake_samples'], axis=None, keep_dims=True)
        fake_max= tf.reduce_max(model_outputs_dict['generator_fake_samples'], axis=None, keep_dims=True)
        fake_0_1=(model_outputs_dict['generator_fake_samples']-fake_min)/(fake_max-fake_min+eps)
        fake_0_255=tf.saturate_cast(fake_0_1*255.0, dtype=tf.uint8)

    postprocessed_outputs={model_head_generator_name:fake_0_255,
                           #model_head_prediction_name:model_outputs_dict['discriminator_decision'],
                           }
    return postprocessed_outputs

def getOptimizer(loss, learning_rate, global_step):
    '''define here the specific optimizer to be used
        for generative approcahes, one use here two optimizer, on for each opponent
        Reminder, model outputs in the the training/val mode:
        {'generator':G, 'D_real_energy':D_real_energy, 'D_fake_energy':D_fake_energy}

    '''
    '''get and reuse the variable 'k' and the real and fake discriminator energy measure from the model graph
    since one cannot retreive them from function parameters here
    HINT: print all available node names:
    '''
    #for tensor in tf.get_default_graph().as_graph_def().node:
    #    print('tensor:'+str(tensor.name))

    with tf.variable_scope("optimizer_adversarial_balancing", reuse=True):
        k=tf.get_variable(name='k')

    G_loss=tf.get_default_graph().get_tensor_by_name('model_loss/Gloss:0')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/BEGAN/G/')
    G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='model/BEGAN/G/')

    D_loss=tf.get_default_graph().get_tensor_by_name('model_loss/Dloss:0')
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/BEGAN/D/')
    D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='model/BEGAN/D/')

    balance=tf.get_default_graph().get_tensor_by_name('model_loss/balance_DG:0')

    if DEBUG_OPTIM is True:
        print('D_vars='+str(D_vars))
        print('G_vars='+str(G_vars))
        print('D_update_ops='+str(D_update_ops))
        print('G_update_ops='+str(G_update_ops))

    # The authors suggest decaying learning rate by 0.5 when the convergence mesure stall
    # carpedm20 decays by 0.5 per 100000 steps
    # Heumi decays by 0.95 per 2000 steps (https://github.com/Heumi/BEGAN-tensorflow/)
    with tf.variable_scope('D_train_op'):
        with tf.control_dependencies(D_update_ops):
            D_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=optimizer_beta1).\
                minimize(D_loss, var_list=D_vars, global_step=global_step)
    with tf.variable_scope('G_train_op'):
        with tf.control_dependencies(G_update_ops):
            G_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=optimizer_beta1).\
                minimize(G_loss, var_list=G_vars, global_step=global_step)

    # It should be ops `define` under control_dependencies
    with tf.control_dependencies([D_train_op,G_train_op]):
        with tf.variable_scope('update_k'):
            training_op = tf.assign(k, tf.clip_by_value(k + lambd_k * balance, 0., 1.)) # define

    if DEBUG_OPTIM is True:
        training_op=tf.Print(training_op, [G_loss, D_loss, k, balance], message='DEBUG [G_loss, D_loss, k, balance]')

    return training_op

def get_total_loss(inputs, model_outputs_dict, labels, weights_loss):
    '''a specific loss for data reconstruction when dealing with autoencoders
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
        weights_loss: the model weights loss that may be used for regularization
    '''
    print('inputs.graph='+str(inputs.graph))
    with tf.variable_scope("optimizer_adversarial_balancing", reuse=False):
        initial_k = tf.constant(0.)
        k=tf.get_variable(name='k', initializer=initial_k, trainable=False)

    D_real_energy=model_outputs_dict['D_real_energy']
    D_fake_energy=model_outputs_dict['D_fake_energy']

    #with tf.variable_scope('D_loss'):
    D_loss = tf.identity(D_real_energy-k*D_fake_energy, name='Dloss')
    G_loss = tf.identity(D_fake_energy, name='Gloss')

    balance = tf.abs(equilibrium_gamma*D_real_energy-D_fake_energy,name='balance_DG')
    convergence_measure = D_real_energy+ balance
    equilibrium=D_fake_energy/D_real_energy #should match hyperparameter equilibrium_gamma
    tf.summary.scalar('G_loss', G_loss),
    tf.summary.scalar('D_loss', D_loss),
    tf.summary.scalar('D_energy/real', D_real_energy),
    tf.summary.scalar('D_energy/fake', D_fake_energy),
    tf.summary.scalar('Equilibrium', equilibrium),
    tf.summary.scalar('convergence_measure', convergence_measure),
    tf.summary.scalar('k', k),

    # sparse-step summary
    G_fake_samples = model_outputs_dict['generator_fake_samples']
    tf.summary.image('fake_sample', G_fake_samples, max_outputs=summary_fake_samples_max_number)
    tf.summary.histogram('G_hist', G_fake_samples) # for checking out of bound
    # histogram all varibles
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name, var)

    return convergence_measure

def get_eval_metric_ops(inputs, model_outputs_dict, labels):
    """Return a dict of the evaluation Ops.
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
    Returns:
        Dict of metric results keyed by name.
    """
    D_real_energy=model_outputs_dict['D_real_energy']
    D_fake_energy=model_outputs_dict['D_fake_energy']
    #FIXME in the paper, equilibrium_gamma is not fixed but is : equilibrium_gamma=E[L(G(z))]/E[L(x)]

    balance = tf.abs(equilibrium_gamma*D_real_energy-D_fake_energy)
    convergence_measure = D_real_energy + balance

    return {
            'model_loss/D_energy/real': tf.metrics.mean_tensor (
                        values=D_real_energy),
            'model_loss/D_energy/fake': tf.metrics.mean_tensor (
                        values=D_fake_energy),
            'model_loss/convergence_measure': tf.metrics.mean_tensor (
                        values=convergence_measure),
            }

'''Define here the input pipelines :
-1. a common function for train and validation modes
-2. a specific one for the serving model_extra_update_ops
'''
def get_input_pipeline_train_val(batch_size, raw_data_files_folder, shuffle_batches):
    """
    FROM : https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0
    Return the input function to get the training data.
    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data (Object): Object holding the loaded mnist data.
    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    class IteratorInitializerHook(tf.train.SessionRunHook):
        """Hook to initialise data iterator after Session is created."""

        def __init__(self):
            super(IteratorInitializerHook, self).__init__()
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            """Initialise the iterator after the session has been created."""
            self.iterator_initializer_func(session)
    iterator_initializer_hook = IteratorInitializerHook()
    # Create a dataset tensor from the images and the labels
    if shuffle_batches is True:#train case
        images=mnist.train.images.reshape([-1, 28, 28, 1])
        labels=mnist.train.labels
    else:#test case
        images=mnist.test.images.reshape([-1, 28, 28, 1])
        labels=mnist.test.labels
    #ensure labels are of type int since the initial uint8 format is not supported for Tensorboard Projector
    labels=labels.astype(np.int32)
    def train_inputs():
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope('Train_val_data'):
            # Get Mnist data
            # Define placeholders
            images_placeholder = tf.placeholder(
                images.dtype, images.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices(
                (images_placeholder, labels_placeholder))
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={images_placeholder: images,
                               labels_placeholder: labels})
            if DEBUG_OPTIM is True:
                next_example=tf.Print(next_example,[tf.reduce_max(next_example), tf.reduce_min(next_example)], message='Input Max, min : ')

            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return train_inputs, iterator_initializer_hook

def get_input_pipeline_train_val_(batch_size, raw_data_files_folder, shuffle_batches):
    ''' define an input pipeline able to load temporal series from a set of
    CSV files and a batch size specified as inputs
    TODO, look at the doc here : https://www.tensorflow.org/programmers_guide/datasets
    @param batch_size : the expected size of a batch
    @param raw_data_files_folder : the folder where CSV files are stored
    @param shuffle_batches : a boolean that activates batch shuffling
    '''
    # Create a dataset tensor from the images and the labels
    if shuffle_batches is True:#train case
        images=mnist.train.images
        labels=mnist.train.labels
    else:#test case
        images=mnist.test.images
        labels=mnist.test.labels

    #create a dataset from the image and related labels arrays
    dataset = tf.contrib.data.Dataset.from_tensor_slices((images, labels))
    # Create batches of data
    dataset = dataset.batch(batch_size)
    # Create an iterator, to go over the dataset
    iterator = dataset.make_initializable_iterator()
    def input_fn():
        raw_batch_img, batch_labels = iterator.get_next()
        batch_img=tf.reshape(raw_batch_img,  shape=[-1, 28, 28, 1])
        tf.Print(batch_img,[tf.max(batch_img), tf.minimum(batch_img)], message='Input Max, min : ')

        return batch_img, tf.cast(batch_labels, tf.int32)
    return input_fn, iterator
'''
################################################################################
## Serving (production) section, define here :
-get_input_pipeline_serving():  the input placeholder of the server that will receive the data
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
'''
serving_input_shape=[1,nb_classes]
def get_input_pipeline_serving():
    '''Build the serving inputs, expecting messages made of :
    -> a batch of size 1 of a single image in the uint8 format (no preliminary normalisation is expected).
    ---> the input is then converted into a float32 4D batch
    '''
    serialized_tf_example = tf.placeholder(
        dtype=tf.float32,
        shape=serving_input_shape,
        name='serialized_input_data')
    print('Served input='+str(serialized_tf_example))
    return tf.estimator.export.ServingInputReceiver(
        serialized_tf_example, {input_data_name: serialized_tf_example})


import cv2
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

    def getInputData(self, idx):
        ''' method that returns data samples complying with the placeholder
        defined in function get_input_pipeline_serving
        Args:
           idx: the input data index
        Returns:
           the data sample with shape and type complying with the server input
        '''
        self.code=np.random.uniform(low=-1.,high=1.,size=serving_input_shape).astype(np.float32)
        return self.code

    def decodeResponse(self, result):
        ''' receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        '''
        response = np.reshape(np.array(result.outputs[served_head].int_val),(28,28)).astype(np.uint8)

        if self.debugMode is True:
            print('Answer shape='+str(response.shape))
        rescaled_sample=cv2.resize(response, (128,128), cv2.INTER_NEAREST)

        cv2.imshow('Generated sample. Press a key to continue...', rescaled_sample)

        cv2.waitKey(1000)

    def finalize(self):
        ''' a function called when the prediction loop ends '''
        print('Prediction process ended successfuly')
