'''
@brief federated MNIST dataset preprocessing for distributed machine learning

Main functions are:
    - the main script, to be used for testing or simply preparing data for the first time    
    - rewrite_dataset: create a tf.dataset for each client and write it in a file
    - load_dataset_info: loads and maybe prepared if not ready train and test tensorflow dataset (tf.data.dataset objects)
    - load_single_dataset_path: get path to the target single tf.data.dataset (train or validation set) 

@author: Mickaël Bettinelli & Alexandre Benoit, LISTIC Lab, France    
'''

import os
import json
import pandas
import subprocess
from collections import defaultdict

def maybe_download_data():
    '''
    try to load the datasets for the federated learning framework
    if datasets are not available then prepare them
    arguments:
        hparams: dictionary of hyperparameters
        batch_size: batch size
        need_resampling: boolean indicating if the data needs to be resampled (default: False)
    returns:
        train_info: dataframe of information about the train dataset : client names and number of samples
        val_info: dataframe of information about the test dataset : client names and number of samples
    '''
    raw_path = "https://raw.githubusercontent.com/MilowB/federated_MNIST_datasets/main/MNIST/" # config1/client1/client1.csv
    raw_path = "https://zenodo.org/record/8094225/files/mnist-data-Federated-Learning.zip" # config1/client1/client1.csv
    dataset_path = os.path.join(os.path.expanduser("~"),'.keras/datasets')
    
    if not(os.path.exists(os.path.join(os.path.expanduser("~"),'.keras/datasets/mnist-data-Federated-Learning.zip'))):
        download_cmd='wget ' + raw_path + ' --directory-prefix '+ dataset_path
        p = subprocess.run(download_cmd, shell=True)
        unzip_cmd = "unzip " + os.path.join(os.path.expanduser("~"),'.keras/datasets/mnist-data-Federated-Learning.zip') + " -d " + dataset_path
        p = subprocess.run(unzip_cmd, shell=True)


# main function for testing and tf.dataset rewriting purposes
if __name__ == '__main__':
    maybe_download_data()
