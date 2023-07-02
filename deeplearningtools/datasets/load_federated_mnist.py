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
    try to download and unzip the demo datasets for the federated learning framework
    '''
    raw_path = "https://zenodo.org/record/8104408/files/mnist-data-federated-learning.zip"
    dataset_path = os.path.join(os.path.expanduser("~"),'.keras/datasets')
    
    if not(os.path.exists(os.path.join(os.path.expanduser("~"),'.keras/datasets/mnist-data-federated-learning.zip'))):
        download_cmd='wget ' + raw_path + ' --directory-prefix '+ dataset_path
        p = subprocess.run(download_cmd, shell=True)
        unzip_cmd = "unzip " + os.path.join(os.path.expanduser("~"),'.keras/datasets/mnist-data-federated-learning.zip') + " -d " + dataset_path
        p = subprocess.run(unzip_cmd, shell=True)


# main function for testing and tf.dataset rewriting purposes
if __name__ == '__main__':
    maybe_download_data()
