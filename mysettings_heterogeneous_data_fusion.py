'''
@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : simple personnal file that defines experiment specific keys to be used with our programs
'''
import DataProvider_input_pipeline
import tensorflow as tf
import numpy as np

#-> set here your own working folder
workingFolder='experiments/heterogeneous_data_embedding'
save_model_variables_to_pandas=True
#-> port number to be used when interracting with the tensorflow-server
tensorflow_server_address='127.0.0.1'
tensorflow_server_port=9000
wait_for_server_ready_int_secs=5
serving_client_timeout_int_secs=1#timeout limit when a client requests a served model

#set here a 'nickname' to your session to help understanding, must be at least an empty string
session_name='recons_xcross_norm'

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
model_file='model_vae_basic.py'
field_of_view=20

display_model_layers_info=False #a flag to enable the display of additionnal console information on the model properties (for debug purpose)
#-> define here a string name used for the train, eval and served models
input_data_name='input'
model_head_embedding_name='code'
model_head_prediction_name=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY#'prediction'
#->define here the output that will be provided by tensorflow-server
served_head=model_head_embedding_name

#-> set the number of summaries store per training epoch (more=more precise BUT higer cost)
nb_summary_per_train_epoch=4

#define image patches extraction parameters
patchSize=224

#random seed used to init weights, etc. Use an integer value to make experiments reproducible
random_seed=None

# learning rate decaying parameters
nbEpoch=50
weights_weight_decay=0.0001
initial_learning_rate=0.001
num_epochs_per_decay=150 #number of epoch keepng the same learning rate
learning_rate_decay_factor=0.1 #factor applied to current learning rate when NUM_EPOCHS_PER_DECAY is reached
predict_using_smoothed_parameters=False#set True to use trained parameters values smoothed (EMA) along the training steps (better results expected)

#set here paths to your data used for train, val
#-> a first set of data
raw_data_dir_train = "/home/alben/workspace/Datasets/Students/train"
raw_data_dir_val = "/home/alben/workspace/Datasets/Students/train"
csv_field_delim=';'
nb_train_samples=756 #manually adjust here the number of temporal items out of the temporal block size
nb_test_samples=756
import pandas
csv_col_names_defaults=pandas.read_csv("/home/alben/workspace/Datasets/Students/defaults.csv", sep=';')
colnames=csv_col_names_defaults.columns
print(colnames)
record_defaults=[]
for col_val in csv_col_names_defaults.loc[0]:
    if np.isnan(col_val):
        record_defaults.append(['NA'])
    else:
        record_defaults.append([1.0])
print('record_defaults'+str(record_defaults))
data_cols=[#{'name':'Sexe','vocabulary_list':["M", "F"]},
            {'name':'Serie/Domaine/Filiere','vocabulary_list':["S", "STI2D", "ES", "L", "P", "STMG", "APU"]},
            {'name':'Niveau_de_la_classe','vocabulary_list':["Faible", "Moyen", "Assez bon", "Bon", "Tres bon"]},
            {'name':'Avis_du_CE','vocabulary_list':["Defavorable", "Reserve", "Favorable", "Tres favorable"]},
            {'name':'Moyenne_Maths', 'normalizer_fn':lambda x:x/20}, #marks are supposed to be out of 20 points and are normalized between 0(0/20) and 1(20/20)
            {'name':'Moyenne_LV1', 'normalizer_fn':lambda x:x/20},
            {'name':'Moyenne_philo-francais', 'normalizer_fn':lambda x:x/20},
            #{'name':'NOTE_DOSSIER'},
            #{'name':'ADMISSIBILITE','vocabulary_list':["BAS DE LISTE", "OUI", "NON"]},
            #{'name':'NOTE_FINALE'},
            ]

label_cols=['Sexe',
            'Nom',
            'Prenom',
            'Numero',
            'Profil_du_candidat',
            'Libelle_etablissement',
            'Departement_etablissement',
            'Serie/Domaine/Filiere',
            'Specialite/Mention/Voie',
            'Serie_diplome',
            'Specialite_diplome',
            'Niveau_de_la_classe',
            'Avis_du_CE',
            'ADMISSIBILITE',
            'NOTE_ENTRETIEN',
            'NOTE_FINALE']#csv_col_names_defaults.drop([data_col['name'] for data_col in data_cols], axis=1)
#prepare a dictionary describing the data and required for the DataProvider_input_pipeline.FileListProcessor_csv_lines function
features_labels={'all_cols':{'names':colnames, 'record_defaults':record_defaults},
                 'data_cols':{'names_opt_categories_or_buckets':data_cols},
                 'labels_cols':{'names':label_cols}}
reference_labels=label_cols#['startDate', 'stopDate'] #to be used if many labels are generated by the get_input_pipeline_train_val function
batch_size=64
raw_data_filename_extension='*.csv'
ref_data_filename_extension='*.csv'

#TODO: have a look here : https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/customestimator/trainer/model.py

####################################################
## Define here use case specific metrics, loss, etc.
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

    # nothing, suppose all feature columns are normalized before
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
                           model_head_prediction_name:model_outputs_dict['reconstructed_data'],
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


    model_gradients=tf.gradients(xs=inputs, ys=model_outputs_dict['code'])[0]

    print('model_gradients='+str(model_gradients))
    transpose_perm=[0]+range(1,len(model_gradients.get_shape()))
    print('transpose_perm='+str(transpose_perm))

    transposed_gradients=tf.transpose(model_gradients, perm=transpose_perm)
    raw_input('transposed_model_gradients='+str(transposed_gradients))

    recon_loss=tf.losses.mean_squared_error(
                                    model_outputs_dict['reconstructed_data'],
                                    inputs,
                                    weights=1.0,
                                    scope=None,
                                    loss_collection=tf.GraphKeys.LOSSES,
                                    #reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
                                    )
    '''
    # E[log P(X|z)]
    recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_outputs_dict['reconstructed_data'],
                                                                        labels=inputs), 1)

    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    kl_loss = 0.5 * tf.reduce_sum(tf.exp(model_outputs_dict['z_logvar']) + model_outputs_dict['z_mu']**2 - 1. - model_outputs_dict['z_logvar'], 1)
    print('dim(Kl)={dimKL}'.format(dimKL=kl_loss.get_shape()))
    print('dim(mu)={dimmu}'.format(dimmu=kl_loss.get_shape()))
    # VAE loss
    vae_loss = tf.reduce_mean(recon_loss + kl_loss)

    return vae_loss#+weights_weight_decay*weights_loss

def get_eval_metric_ops(inputs, model_outputs_dict, labels):
    '''Return a dict of the evaluation Ops.
    Args:
        inputs: the input data samples batch
        model_outputs_dict: the dictionnay of model outputs, field names must comply withthe ones defined in the model_file
        labels: the reference data / ground truth if available
    Returns:
        Dict of metric results keyed by name.
    '''
    return {
            'MSE': tf.metrics.mean_squared_error(
                labels=inputs,
                predictions=model_outputs_dict['reconstructed_data'],
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
    #load all csv files to use for training
    raw_data_files=DataProvider_input_pipeline.extractFilenames(root_dir=raw_data_files_folder, file_extension=raw_data_filename_extension)
    print('Input files found='+str(raw_data_files))

    class IteratorInitializerHook(tf.train.SessionRunHook):
        """Hook to initialise data iterator after Session is created."""

        def __init__(self):
            super(IteratorInitializerHook, self).__init__()
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            """Initialise the iterator after the session has been created."""
            self.iterator_initializer_func(session)
    iterator_initializer_hook = IteratorInitializerHook()
    def input_fn():
        with tf.name_scope("retrieve_data"):
            dataset=DataProvider_input_pipeline.FileListProcessor_csv_lines(
                                                                                files=raw_data_files,
                                                                                csv_field_delim=csv_field_delim,
                                                                                queue_capacity=batch_size*5,
                                                                                shuffle_batches=shuffle_batches,
                                                                                batch_size=batch_size,
                                                                                features_labels=features_labels,
                                                                                device="/cpu:0",
                                                                                debug=False)
        #get an iterator on the dataset
        iterator = dataset.make_initializable_iterator()
        # Set runhook to initialize the iterator
        iterator_initializer_hook.iterator_initializer_func = \
            lambda sess: sess.run(iterator.initializer)

        #get a new data batch
        data_labels_batch = iterator.get_next()

        #print('data_labels_batch='+str(data_labels_batch))
        #separate input data and labels
        data_cols={}
        for data_col in features_labels['data_cols']['names_opt_categories_or_buckets']:
            data_cols[data_col['name']]=data_labels_batch[data_col['name']]
        print('**************data_cols='+str(data_cols))
        #labels must be of the same type => here, converting to string and stack them
        col_str_list=[]
        for label_col in features_labels['labels_cols']['names']:
            #print('Considering label:'+str(data_labels_batch[label_col]))
            if data_labels_batch[label_col].dtype.name != 'string':
                col_str_list.append(tf.as_string(data_labels_batch[label_col]))
            else:
                col_str_list.append(data_labels_batch[label_col])
        labels_cols_str=tf.stack(col_str_list, 1)
        #print('Labels cols='+str(labels_cols_str))
        #raw_input('timestamps_start_stop='+str(timestamps_start_stop))
        return data_cols, labels_cols_str
    return input_fn, iterator_initializer_hook

def features_dict_to_tensor(features_dict):
    ''' function that converts input data columns into a dense tensor in the
    appropriate way using tf.feature_column converters
    NOTE that the features_labels variable may include normalization functions for some feature columns
    @param features_dict: the input data dictionnary
    @return a dense tensor in the tf.float32 format
    '''
    dense_input=DataProvider_input_pipeline.extract_feature_columns(features_dict, features_labels)
    #dense_input=tf.Print(dense_input, [dense_input], message='Dense Input Data=', summarize=dense_input.get_shape().as_list()[1])
    return dense_input

'''
################################################################################
## Serving (production) section, define here :
-get_input_pipeline_serving():  the input placeholder of the server that will receive the data
-class Client_IO, a class to specifiy input data requests and response on the client side when serving a model
For performance/enhancement of the model, have a look here for graph optimization: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/tools/graph_transforms/README.md
'''
def get_input_pipeline_serving():
    """Build the serving inputs."""
    csv_row = tf.placeholder(
      shape=[None],
      dtype=tf.string
    )
    parsed_line = tf.decode_csv(csv_row, record_defaults=features_labels['all_cols']['record_defaults'], field_delim=csv_field_delim)
    features = dict(zip(features_labels['all_cols']['names'], parsed_line))

    unused_cols = set(features_labels['all_cols']['names']) - {col['name'] for col in features_labels['data_cols']['names_opt_categories_or_buckets']} \
                                                            - {col for col in features_labels['labels_cols']['names']}
    # Remove unused columns
    for col in unused_cols:
        features.pop(col)
    return tf.estimator.export.ServingInputReceiver(
      features, {'csv_row': csv_row})

    '''Build the serving inputs, expecting messages made of :
    -> a batch of size 1.
    -> a data buffer of type float32 of the same shape as each of the elements used along training (no preliminary normalisation is expected)
    '''
    serialized_tf_example = tf.placeholder(
        dtype=tf.string,
        shape=[1, temporal_series_length, len(record_defaults)-1],
        name='serialized_input_data')

    return tf.estimator.export.ServingInputReceiver(
        serialized_tf_example, {input_data_name: serialized_tf_example})

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
        #here, only random numbers
        sample=np.random.random([1,240,12]).astype(np.float32)
        if self.debugMode is True:
            print('Generating input features (random values) of shape '+str(sample.shape))
        return sample

    def decodeResponse(self, result):
        ''' receive the server response and decode as requested
            have a look here for data types : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
            have a look at gRPC error codes here : https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            Args:
            result: a PredictResponse object that contains the request result
        '''
        response = np.array(result.outputs[served_head].float_val)
        print('Answer shape='+str(response.shape))

    def finalize(self):
        ''' a function called when the prediction loop ends '''
        print('Prediction process ended successfuly')
