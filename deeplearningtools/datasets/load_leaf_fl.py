'''
@brief LEAF benchmark dataset preprocessing for centralized or distributed machine learning

Main functions are:
    - the main script, to be used for testing or simply preparing data for the first time    - rewrite_dataset: create a tf.dataset for each client and write it in a file
    - load_dataset_info: loads and maybe prepared if not ready train and test tensorflow dataset (tf.data.dataset objects)
    - load_single_dataset_path: get path to the target single tf.data.dataset (train or validation set) 

@author: Alexandre Benoit, LISTIC Lab, France    
'''

import os
import json
import pandas
import subprocess
from collections import defaultdict
import tensorflow as tf

def maybe_download_data(hparams, min_batch_size, fraction_data=1.0, need_resampling=False):
    '''
    download the dataset if necessary from the https://github.com/TalwalkarLab/leaf.git repository
    arguments:
        hparams: dictionary of hyperparameters that contains the following keys:
            data: name of the dataset
            isIID: boolean indicating if the dataset is IID or not
            minCl: minimum number of clients
        fraction_data: fraction of the data to be used (1.0 means all the data)
        min_batch_size: minimum batch size used in the training/val process, will be used to remove too small clients.
        need_resampling: boolean indicating if the dataset needs to be resampled (True) or not (False)
    returns:
        nothing
    '''
    #first download the dataset if necessary
    dataset_path=os.path.join(os.path.expanduser("~"),'LEAFdataset')
    raw_data_already_downloaded=os.path.exists(dataset_path)
    if not(raw_data_already_downloaded):
        os.makedirs(dataset_path)
        _GIT_repo='https://github.com/TalwalkarLab/leaf.git'
        print('Cloning the LEAF repo',_GIT_repo, ' and preparing data... please be patient') 
        initial_path=os.getcwd()
        os.chdir(dataset_path)
        clone_cmd='git clone '+_GIT_repo
        p = subprocess.run(clone_cmd, shell=True)
        print('LEAF git repository cloned in ', dataset_path)
        print('Preprocessing dataset ', hparams['data'])

        need_resampling=True # one then needs to preprocess data

    if need_resampling:
        print('Resampling/preprocssing data with the LEAF benchmark tools...')
        if os.path.exists(os.path.join(dataset_path,'leaf','data',hparams['data'], 'data', 'train')):
            print('Data samples already prepared, removing old samples...')
            print('TODO/INFO: this code can be optimized to first check if compliant preprocessed samples are already there thus avoiding some long processing...')
            remove_old_sampling_cmd='rm -rf ~/LEAFdataset/leaf/data/{dataset}/data/sampled_data ~/LEAFdataset/leaf/data/{dataset}/data/test ~/LEAFdataset/leaf/data/{dataset}/data/train ~/LEAFdataset/leaf/data/{dataset}/data/rem_user_data/ ~/LEAFdataset/leaf/data/{dataset}/data/sampled_data/'.format(dataset=hparams['data'])
            p = subprocess.run(remove_old_sampling_cmd, shell=True)
        os.chdir(os.path.join(dataset_path,'leaf','data',hparams['data']))
        sampling='iid'
        if hparams['isIID']==False:
            sampling='niid'
        # this code does not impose the number of clients, all compatible ones are prepared
        data_preprocess_cmd='./preprocess.sh -s {sampling} --sf {frac} -k {batch} -t sample'.format(sampling=sampling, frac=fraction_data, batch=min_batch_size)
        # REMINDER femnist small ./preprocess.sh -s niid --iu minCl --sf 0.05 -k 0 -t sample
        print('running command:', data_preprocess_cmd)
        p = subprocess.run(data_preprocess_cmd, shell=True)
        print('Data preprocessing ok')

#read the data
def read_dir(data_dir):
    '''
    from LEAF benchmark original codes : leaf/models/utils/model_utils.py
    parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        data: dictionary of related data
    '''
    print('Reading json files from:', data_dir)
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    print('Found files', files)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    print('Loaded data for', len(clients), 'clients')
    return clients, groups, data

def create_write_single_client_tf_dataset(original_dataset_client_data, path, client_id):
    '''
    create a tf.dataset for a single client and write it in a file
    arguments:
        original_dataset_dict: dictionary of the original dataset
        path: path where to write the tf.dataset
        client_id: id of the client
    returns:
        nothing but writes a tensorflow dataset in path/client_id
    '''
    print('client_id', client_id, '-> number of samples:', len(original_dataset_client_data['y']))
    dataset=tf.data.Dataset.from_tensor_slices(original_dataset_client_data)
    cli_path=os.path.join(path, client_id)
    dataset.save(path=cli_path, compression='GZIP')

def rewrite_dataset(original_dataset_dict:dict, path:str):
    ''' 
    create a tf.dataset for each client and write it in a file
    arguments:
        original_dataset_dict: dictionary of the original dataset
        path: path where to write the tf.dataset
    returns:
        nothing
    '''
    #prepare dataset files location
    if not(os.path.exists(path)):
        os.makedirs(path)
    print('Loading single client data and preparing related tensorflow datasets...')
    # write a csv file with the list of client ids
    clients_info={'client_names':[], 'samples':[]}
    print('original_dataset_dict.keys():', original_dataset_dict.keys())
    for client_id in original_dataset_dict.keys():
        clients_info['client_names'].append(client_id.split('_')[0]) #remove the suffix _xxx to allow for same train/test sorting results
        clients_info['samples'].append(len(original_dataset_dict[client_id]['y']))
    print('clients_info', clients_info)
    clients_info=pandas.DataFrame.from_dict(clients_info)
    clients_info.to_csv(os.path.join(path,'clients_info.csv'))
    nb_datasets=len(original_dataset_dict.keys())
    for id, client_id in enumerate(original_dataset_dict.keys()):
        print('preparing dataset ', id, '/', nb_datasets)
        ''' write the dataset in the format expected by the federated learning framework '''
        create_write_single_client_tf_dataset(original_dataset_dict[client_id], client_id=client_id.split('_')[0], path=path)
    
    return clients_info

def prepare_datasets(hparams, batch_size, need_resampling=False):
    ''' 
    prepare the datasets for the federated learning framework
    arguments:
        hparams: dictionary of hyperparameters
        batch_size: batch size
        need_resampling: boolean indicating if the data needs to be resampled (default: False)
    returns:
        train_info: dataframe of information about the train dataset : client names and number of samples
        val_info:   dataframe of information about the test dataset : client names and number of samples
    '''
    #get data if necessary
    maybe_download_data(hparams, min_batch_size=batch_size, need_resampling=need_resampling)

    # load train data
    raw_data_dir_train = os.path.join(os.path.expanduser("~"),'LEAFdataset', 'leaf', 'data', hparams['data'], 'data', 'train')
    raw_data_dir_test =   os.path.join(os.path.expanduser("~"),'LEAFdataset', 'leaf', 'data', hparams['data'], 'data', 'test')

    print('******* Loading train data... memory requirements may be high')
    clients, groups, data = read_dir(raw_data_dir_train)
    #rewrite the dataset in tf.dataset format
    print('rewriting train data')
    train_info=rewrite_dataset(data,
                               os.path.join(os.path.expanduser("~"),'LEAFdataset', 'leaf', 'data', hparams['data'], 'data', 'train', 'tf_datasets'),
                               )
    print('train data rewritten')

    # load val data
    print('******* Loading validation data... memory requirements may be high')
    clients, groups, data = read_dir(raw_data_dir_test)
    if len(clients)<hparams['minCl']:
        raise ValueError('dataset preprocessing does not provide enough clients compared to the expected number :hparams[minCl]=', hparams['minCl'])
    #rewrite the dataset in tf.dataset format
    print('rewriting test data')
    test_info=rewrite_dataset(data,
                             os.path.join(os.path.expanduser("~"),'LEAFdataset', 'leaf', 'data', hparams['data'], 'data', 'test', 'tf_datasets'),
                             )
    print('test data rewritten')

    return train_info, test_info

def load_dataset_info(hparams, batch_size, need_resampling=False):
    '''
    try to load the datasets information for the federated learning framework
    if datasets are not available then prepare them
    arguments:
        hparams: dictionary of hyperparameters
        batch_size: batch size
        need_resampling: boolean indicating if the data needs to be resampled (default: False)
    returns:
        train_info: dataframe of information about the train dataset : client names and number of samples
        val_info: dataframe of information about the test dataset : client names and number of samples

    '''
    if need_resampling or not(os.path.exists(os.path.join(os.path.expanduser("~"),'LEAFdataset', 'leaf', 'data', hparams['data'], 'data', 'test', 'tf_datasets', 'clients_info.csv'))):
        return prepare_datasets(hparams, batch_size, need_resampling)
    # load train data
    train_info = pandas.read_csv(os.path.join(os.path.expanduser("~"),'LEAFdataset', 'leaf', 'data', hparams['data'], 'data', 'train', 'tf_datasets', 'clients_info.csv')).sort_values(by=['client_names'])
    # load val data
    test_info = pandas.read_csv(os.path.join(os.path.expanduser("~"),'LEAFdataset', 'leaf', 'data', hparams['data'], 'data', 'test', 'tf_datasets', 'clients_info.csv')).sort_values(by=['client_names'])
    return train_info, test_info

def load_single_dataset_path(datasetname:str, target_client_name:str, is_train:bool):
    '''
    load the path of a single dataset according to the target client name and experiment hyperparameters (key data that should contain either celeba or femnist values)
    arguments:
        datasetname: the name of the dataset, currently: 'celeba' or 'femnist'
        target_client_name: name of the client to load (string that should be in the list of clients listed by the load_dataset_info function)))
        is_train: boolean indicating if the dataset is a train dataset (True) or a test dataset (False)
    '''
    if not(os.path.exists(os.path.join(os.path.expanduser("~"),'LEAFdataset', 'leaf', 'data', datasetname, 'data', 'test', 'tf_datasets', 'clients_info.csv'))):
        raise ValueError('dataset seems to be not available, please run load_dataset_info before')
    split='train' if is_train else 'test'
    return os.path.join(os.path.expanduser("~"),'LEAFdataset', 'leaf', 'data', datasetname, 'data', split, 'tf_datasets', target_client_name)

# main function for testing and tf.dataset rewriting purposes
if __name__ == '__main__':

    hparams={'isIID':False,
             'data':'celeba', #choose among 'femnist' and 'celeba'
             'isIID':False,#set False to get non iid sampling other clients, True for iid
             'procID':0,
             'minCl':200,
             }
    min_batch_size=10
    print('Preparing the femnist dataset from LEAF benchmark raw data with the following setup:', hparams)    
    train_info, test_info=prepare_datasets(hparams, min_batch_size, need_resampling=True)
    print('Data prepared')
    print('train_info', train_info)
    print('val_info', test_info)
