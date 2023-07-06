# ========================================
# FileName: start_kafka_producer.py
# Date: 29 june 2020 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of tools to help running a kafka log server
# for DeepLearningTools.
# =========================================
"""
Standardized script that loads an experiment in order to use the 'get_input_pipeline' function as a data provider to a kafka log queue

use example:
-> writing to queue 'samples' on server ''

# tested with kafka install and config:
wget  https://downloads.apache.org/kafka/2.7.1/kafka_2.13-2.7.1.tgz
tar -xzf kafka_2.13-2.7.1.tgz

#-> start zookeeper and kafka
./kafka_2.13-2.7.1/bin/zookeeper-server-start.sh -daemon ./kafka_2.13-2.7.1/config/zookeeper.properties
./kafka_2.13-2.7.1/bin/kafka-server-start.sh -daemon ./kafka_2.13-2.7.1/config/server.properties
echo "Waiting for 10 secs until kafka and zookeeper services are up and running"
sleep 10

#-> create the 'demo-pics' topic, no worry, this script creates its own topics, relying on the experiments config

#-> check topic behaviors (ex: demo-pics)
./kafka_2.13-2.7.1/bin/kafka-topics.sh --describe --bootstrap-server 127.0.0.1:9092 --topic demo-pics

#-> not forget to delete topic to keep disk space...
./kafka_2.13-2.7.1/bin/kafka-topics.sh  --bootstrap-server 127.0.0.1:9092 --delete --topic demo-pics

#-> list all available topics on a server:
./kafka_2.13-2.7.1/bin/kafka-topics.sh  --list --bootstrap-server 127.0.0.1:9092

#-> get topic details:
./kafka_2.13-2.7.1/bin/kafka-topics.sh  --bootstrap-server=localhost:9092 --describe --topic demo-pics

# LOGS LOCATION : by default, queues/logs are stored here : /tmp/kafka-logs
"""

import argparse
import numpy as np

import deeplearningtools.helpers.kafka_io
import deeplearningtools.experiments_manager
import deeplearningtools.helpers.tensor_msg_io
import deeplearningtools.tools.experiments_settings_surgery

DEFAULT_LOG_QUEUE_NAME='default_queue'

def get_commands():
    """
    Defines the command line argument parser dedicated to the running session.
    """
    argparser = argparse.ArgumentParser(description='Data producer to Kafka that makes use of DeepLearningTools experiments input data pipelines')
    argparser.add_argument("-u","--usersettings",
                        help="filename of the settings file that defines an experiment")
    argparser.add_argument("-v","--isvalidationdata", action='store_true',
                        help="by default, this script will push training data but if this option is used, validation data will be pushed instead")
    argparser.add_argument("-s","--server", default=['127.0.0.1:9092'],
                        help="specify the kafka servers where to push the data")
    argparser.add_argument("-q","--overridequeue", default=DEFAULT_LOG_QUEUE_NAME,
                        help="specify the kafka log queue where to push the data")
    argparser.add_argument("-pid","--procID", default=None,
                        help="Specifiy here an ID to identify the process (useful for federated training sessions)")
    argparser.add_argument("-e","--epochs", default=1, type=int,
                        help="customize the number of data epoch to be produced and stored, default is one epoch, more usually involoves more data augmentation")

    argparser.add_argument("-c","--check", action='store_true',
                        help="does not produce but simply reads an existing kafka dataset")
    return argparser

def run(commands):
    """
    Main function that applies runs kafka library to push/read data to/from a kafka server.
    
    :param commands: The decoded expected flags defined in the get_commands() function.
         
    """
    #get experiment settings filename path
    settings_file=commands.usersettings

    #take into account some hyperparameters if required
    external_hparams={}
    if commands.procID is not None:
        external_hparams['procID']=commands.procID

    if len(external_hparams.keys())>0:
        settings_file=deeplearningtools.tools.experiments_settings_surgery.insert_additionnal_hparams(settings_file, external_hparams)

    #load experiment file
    usersettings, sessionFolder = deeplearningtools.experiments_manager.loadExperimentsSettings(settings_file, isServingModel=False)

    # get dataset
    dataset_folder=None
    if commands.isvalidationdata:
        dataset_folder=usersettings.raw_data_dir_val
    else:
        dataset_folder=usersettings.raw_data_dir_train
    dataset =usersettings.get_input_pipeline(raw_data_files_folder=dataset_folder,
                                            isTraining=not(commands.isvalidationdata),
                                            batch_size=1,
                                            nbEpoch=commands.epochs)

    # push data to the kafka server
    if commands.overridequeue == DEFAULT_LOG_QUEUE_NAME: # if queue name is not overwriten, create a queue name relying on experiment name+procID when applicable
        log_queue_name=usersettings.session_name
        # in case the procID exists, add this to the queue name

        if 'procID' in usersettings.hparams.keys():
            log_queue_name+=str(usersettings.hparams['procID'])
        
        if commands.isvalidationdata:
            log_queue_name+='val'
        else:
            log_queue_name+='train'

    if commands.check is False:
        print('**** Pushing data on kafka log queue', log_queue_name)
        kafka_writer=deeplearningtools.helpers.kafka_io.KafkaIO(log_queue_name, commands.server)
        kafka_writer.kafka_producer_tf(dataset, log='log_queue_name.info')
    else:
        print('*** reading from existing queue, log_queue_name')
        train_val_dataset_features=deeplearningtools.helpers.tensor_msg_io.get_data_label_features_from_dataset(dataset)
        kafka_dataset_reader=deeplearningtools.helpers.kafka_io.KafkaIO(topic_name=log_queue_name, bootstrap_servers=usersettings.kafka_bootstrap_servers, element_spec=dataset.element_spec)
        dataset=kafka_dataset_reader.kafka_dataset_consumer_tf_custom(train_val_dataset_features, batch_size=usersettings.batch_size, shuffle=False)
        
        # to be comparable with the original dataset:
        dataset_orig =usersettings.get_input_pipeline(  raw_data_files_folder=dataset_folder,
                                                        isTraining=not(commands.isvalidationdata),
                                                        batch_size=usersettings.batch_size,
                                                        nbEpoch=1)

        '''for id, sample_duo in enumerate(datasetzip(dataset, dataset_orig)):
            sample=sample_duo[0]
            sample_orig=sample_duo[1]
            print('### KAFKA sample:', id)
            print('sample:',sample)
            print('### ORIG. sample:', id)
            print('sample:',sample_orig[0])
        '''
        for id, sample in enumerate(dataset):
            print('### KAFKA sample:', id)
            print('sample:',sample)
        print('Read all the dataset with number of samples:', id)

if __name__ == "__main__":

    # retreive command line arguents
    parser = get_commands()
    FLAGS=parser.parse_args()
    run(FLAGS)

