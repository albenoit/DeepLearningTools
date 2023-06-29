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
    configs = ["config1/", "config2/", "config3/"]
    clients = ["client" + str(i) for i in range(1, 11)]

    dataset_path = os.path.join(os.path.expanduser("~"),'.keras/datasets/federated_mnist')
    # check if 1 csv of the datasets exists. If it does not, other csv files probably don't exist too, so we download the whole dataset
    for config in configs:
        for client in clients:
            target_file=os.path.join(dataset_path, config, client+".csv")
            print('looking for file', target_file)
            if not(os.path.exists(target_file)):
    
                download_cmd='wget ' + raw_path + config + client + "/" + client + ".csv" + ' --directory-prefix '+ dataset_path + "/" + config
                p = subprocess.run(download_cmd, shell=True)
                client_number = int(client.split("t")[1])
                new_number = "client" + str(client_number - 1)
                mv_cmd = 'mv ' + dataset_path + "/" + config + client + ".csv" + " " + dataset_path + "/" + config + new_number + ".csv"
                p = subprocess.run(mv_cmd, shell=True)

    target_file=os.path.join(dataset_path, "mnist_test.csv")
    print('looking for test file', target_file)
    if not(os.path.exists(target_file)):
        download_cmd='wget ' + raw_path + "/mnist_test.csv" + ' --directory-prefix '+ dataset_path + "/"
        p = subprocess.run(download_cmd, shell=True)



# main function for testing and tf.dataset rewriting purposes
if __name__ == '__main__':
    maybe_download_data()