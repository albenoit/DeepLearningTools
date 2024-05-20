""" Inspired by https://temporian.readthedocs.io/en/stable/tutorials/anomaly_detection_unsupervised/#installation-and-imports

Loading the OmnyAnomaly dataset from the server.
"""

import os
import urllib.request
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import temporian as tp
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

"""
The dataset is made of 3 groups of 8, 9, and 11 machines respectively, with names "machine-1-1", ..., "machine-3-11". Let's list the machine names, and download records locally.
"""

# -> define general data configuration
ALL_MACHINES_PER_GROUP = [8, 9, 11]  # Full dataset
DATA_DIR = Path("datasamples/temporian_server_machine_dataset")
CLIENT_INFO_FILENAME = "clients_info.csv"
DROP_COLS = ["machine"]#, "timestamp"]
DATA_SPLIT_NAMES=["train", "val"]

def get_data_path():
    prefix=''
    if 'experiments' in os.getcwd():
        prefix=os.getcwd().split("experiments")[0]
    train_path=os.path.join(prefix, DATA_DIR, DATA_SPLIT_NAMES[0]+CLIENT_INFO_FILENAME)
    test_path=os.path.join(prefix, DATA_DIR, DATA_SPLIT_NAMES[1]+CLIENT_INFO_FILENAME)
    data_base_dir=os.path.join(prefix, DATA_DIR)

    return train_path, test_path, data_base_dir

def get_machine_names(machines_per_group: List[int]):
    """Generate the list of machine names from the number of machines per group.

    Args:
        machines_per_group (List[int]): a list of machine number per known groups of machines (3 groups)

    Returns:
        List[str]: a list of machine names
    """
    print(
        "machines_per_group", machines_per_group, "full_dataset", ALL_MACHINES_PER_GROUP
    )
    if len(machines_per_group) != 3:
        raise ValueError("machines_per_group must have 3 elements")
    if not all(isinstance(x, int) for x in machines_per_group):
        raise ValueError("machines_per_group must contain only integers")
    if not all(
        (x > 0 and x <= max)
        for x, max in zip(machines_per_group, ALL_MACHINES_PER_GROUP)
    ):
        raise ValueError("machines_per_group must contain only positive integers")
    machines = [
        f"machine-{group}-{id}"
        for group, machine in zip(range(1, 4), machines_per_group)
        for id in range(1, machine + 1)
    ]
    return machines


def get_single_machine_name(group: int, id: int):
    """Generate the name of a single machine from its group and id.

    Args:
        group (int): the group number of the machine
        id (int): the id number of the machine

    Returns:
        str: the name of the machine
    """
    return f"machine-{group}-{id}"


def maybe_download_data():
    """Download (if necessary) the data and labels of the machines from the server."""
    dataset_url = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
    machines = get_machine_names(ALL_MACHINES_PER_GROUP)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download the data and labels for each machine to its own folder
    for machine in machines:
        print(f"Download data of {machine}")

        machine_dir = DATA_DIR / machine
        machine_dir.mkdir(exist_ok=True)

        data_path = machine_dir / "data.csv"
        if not data_path.exists():
            urllib.request.urlretrieve(f"{dataset_url}/test/{machine}.txt", data_path)

        labels_path = machine_dir / "labels.csv"
        if not labels_path.exists():
            urllib.request.urlretrieve(
                f"{dataset_url}/test_label/{machine}.txt", labels_path
            )


# Loading the data
def load_csv_to_dataframes(machines: List[str]) -> List[pd.DataFrame]:
    """Load the data and labels of the machines from csv files and return them as a list of pandas dataframes.

    Args:
        data_dir (Path): path to the data directory
        machines (List[str]): a list of machine number per known groups of machines (3 groups)

    Returns:
        List[pd.DataFrame]: a list of pandas dataframes
    """
    dataframes = []
    clients_info = {"client_names": [], "samples": []}
    print("Loading data from machines...", machines)
    for machine in machines:
        machine_dir = DATA_DIR / machine

        # Read the data and labels
        print(f"Load data of {machine}...", end="")
        df = pd.read_csv(machine_dir / "data.csv", header=None).add_prefix("f")
        labels = pd.read_csv(machine_dir / "labels.csv", header=None)
        df = df.assign(label=labels)
        df["machine"] = machine
        df["timestamp"] = range(df.shape[0])
        dataframes.append(df)
        n_events = df.shape[0]
        print(f"found {n_events} events")
        clients_info["client_names"].append(
            machine
        )  # remove the suffix _xxx to allow for same train/test sorting results
        clients_info["samples"].append(n_events)
    # print('clients_info', clients_info)
    clients_info = pd.DataFrame.from_dict(clients_info)
    clients_info.to_csv(os.path.join(DATA_DIR, CLIENT_INFO_FILENAME))

    return dataframes


# Convert the dataframes into a single Temporian EventSet
def get_eventset(dataframes: List[pd.DataFrame]) -> tp.EventSet:
    """From a list of pandas dataframes, create a temporian EventSet.

    Args:
        dataframes (List[pd.DataFrame]): the list of dataframes, each one representing a machine.

    Returns:
        tp.EventSet: _description_
    """
    evset = tp.combine(*map(tp.from_pandas, dataframes))

    # Index the EventSet according the the machine name.
    evset = evset.set_index("machine")

    # Cast the feature and label to a smaller dtypes to same one memory.
    evset = evset.cast(tp.float32).cast({"label": tp.int32})

    return evset

def load_omny_server_data_anomaly(machines: List[str]) -> tp.EventSet:
    """Load the OmnyAnomaly dataset from the server ad return data as a temporian EventSet.

    Args:
        data_dir (Path): path to the data directory
        machines (List[str]): a list of machine number per known groups of machines (3 groups)

    Returns:
        tp.EventSet: an agregation of the machine data
    """
    maybe_download_data()
    print("target machines:", machines)
    dataframes = load_csv_to_dataframes(machines)
    evset = get_eventset(dataframes)

    return evset

def split_event_set(evset: tp.EventSet, test_frac: float = 0.2):

    # Average length of the records
    average_duration = np.mean([len(machine_data.timestamps) for _, machine_data in evset.data.items()])
    print("average_duration:", average_duration)

    # Select the train/test cutoff.
    # Note: All the machines are cut at the same time. This way, we can apply pre-processing that
    # exchange data between the machines without risk of label leakage!
    train_cutoff = average_duration * (1 - test_frac)
    print("train_cutoff:", train_cutoff)

    # Compute masks and split data based on cutoff
    train_mask = evset.timestamps() <= int(train_cutoff)
    test_mask = ~train_mask

    # Split EventSets
    train_evset = evset.filter(train_mask)
    test_evset = evset.filter(test_mask)

    print(f"Train events: {train_evset.num_events()}")
    print(f"Test events: {test_evset.num_events()}")

    return train_evset, test_evset

def evsets_to_csv_per_machine(data_train: tp.EventSet, data_test: tp.EventSet) -> tuple:
    """
    Converts event sets to CSV files per machine and saves them.

    Args:
        data_train (tp.EventSet): The training data set.
        data_test (tp.EventSet): The testing data set.

    Returns:
        tuple: A tuple containing the following:
            - train_info (pd.DataFrame): Information about the training data set.
            - test_info (pd.DataFrame): Information about the testing data set.
            - global_max_min (pd.DataFrame): Maximum and Minimum values for each feature across all machines.
    """
    # Save the data to CSV files
    train_info = {"client_names": [], "samples": []}
    test_info = {"client_names": [], "samples": []}
    feature_names = data_train.schema.feature_names()
    global_max = pd.DataFrame(columns=feature_names)
    global_min = pd.DataFrame(columns=feature_names)
    # get machine names and number of samples
    for set_name, data_split, info in zip(DATA_SPLIT_NAMES, [data_train, data_test], [train_info, test_info]):
        set_path = os.path.join(DATA_DIR, set_name)
        os.makedirs(set_path, exist_ok=True)
        for machine_id, machine_data in data_split.data.items():
            machine = machine_id[0].decode('utf-8')
            data_dict= {'timestamps':machine_data.timestamps}
            for name, feat in zip(feature_names, machine_data.features):
                data_dict[name] = feat
            #print("************************\n==> ", machine + ' data_dict:', data_dict)
            df = pd.DataFrame.from_dict(data_dict)
            if set_name == DATA_SPLIT_NAMES[0]:
                # append max values to global_max as new lines
                #create a dataframe from the max values of the machine and keep machine name as index
                global_max = pd.concat([global_max,df.max().to_frame(name=machine).T], axis=0)
                global_min = pd.concat([global_min,df.min().to_frame(name=machine).T], axis=0)
            nb_samples = len(machine_data.timestamps)
            info["client_names"].append(machine)
            info["samples"].append(nb_samples)
            single_machine_path = os.path.join(set_path, f"{machine}.csv")
            df.to_csv(single_machine_path, index=False)
    print("train_info", train_info)
    print("test_info", test_info)
    train_info = pd.DataFrame.from_dict(train_info)
    test_info = pd.DataFrame.from_dict(test_info)
    train_path, test_path, data_base_dir = get_data_path()
    train_info.to_csv(train_path, index=False
    )
    test_info.to_csv(test_path, index=False
    )
    #finalise min and max: keep min and max values for each feature
    global_max_min=pd.concat([global_min.min(), global_max.max()], axis=1)
    global_max_min.columns=['min', 'max']
    global_max_min.to_csv(os.path.join(data_base_dir, 'min_max.csv'), index=True)
    return train_info, test_info, global_max_min

def prepare_datasets(train_split: float = 0.8):
    """
    prepare the datasets for the federated learning framework
    arguments:
        hparams: dictionary of hyperparameters
        batch_size: batch size
        need_resampling: boolean indicating if the data needs to be resampled (default: False)
    returns:
        tuple: A tuple containing the following:
            - train_info (pd.DataFrame): Information about the training data set.
            - test_info (pd.DataFrame): Information about the testing data set.
            - global_max_min (pd.DataFrame): Maximum and Minimum values for each feature across all machines.
    """
    # load the data
    evset = load_omny_server_data_anomaly(machines=get_machine_names(ALL_MACHINES_PER_GROUP))
    
    # split the data into train and test
    data_train, data_test = split_event_set(evset, test_frac=1-train_split)

    train_info, test_info, global_max_min = evsets_to_csv_per_machine(data_train, data_test)

    return train_info, test_info, global_max_min

def load_dataset_info(hparams:dict, need_resampling=False):
    """
    try to load the datasets for the federated learning framework
    if datasets are not available then prepare them
    arguments:
        hparams (dict): dictionary of hyperparameters
        need_resampling (bool): boolean indicating if the data needs to be resampled (default: False)
    returns:
        train_info: dataframe of information about the train dataset : client names and number of samples
        val_info: dataframe of information about the test dataset : client names and number of samples
        global_max_min: dataframe of maximum and minimum values for each feature across all machines
        root_dir: root directory of the data
    """

    train_path, test_path, data_base_dir = get_data_path()
    if need_resampling or not (os.path.exists(train_path) and os.path.exists(test_path)):
        print('need_resampling', need_resampling)
        print('train_path', train_path, 'test_path', test_path)
        print(os.getcwd())
        print("Need to prepare datasets...")
        prepare_datasets(train_split=hparams["trainSplit"])
    print('Data ready in', train_path, test_path)

    # load train data
    train_info = pd.read_csv(train_path).sort_values(by=["client_names"])
    test_info = pd.read_csv(test_path).sort_values(by=["client_names"])
    print('data_base_dir',data_base_dir)
    global_max_min = pd.read_csv(os.path.join(data_base_dir, "min_max.csv"), index_col=0)
    return train_info, test_info, global_max_min, data_base_dir

def load_single_dataset_path(target_client_name: str, is_train: bool):
    """
    load the path of a single dataset according to the target client name and experiment hyperparameters (key data that should contain either celeba or femnist values)
    arguments:
        target_client_name: name of the client to load (string that should be in the list of clients listed by the load_dataset_info function)))
        is_train: boolean indicating if the dataset is a train dataset (True) or a test dataset (False)
    """
    if not (
        os.path.exists(
            os.path.join(os.path.expanduser("~"), ".keras/datasets/adult/adult.csv")
        )
    ):
        raise ValueError(
            "dataset seems to be not available, please run load_dataset_info before"
        )
    split = DATA_SPLIT_NAMES[0] if is_train else DATA_SPLIT_NAMES[1]
    return os.path.join(DATA_DIR, split, f"{target_client_name}.csv")
    


# main function for testing/demo
if __name__ == "__main__":

    machines_per_group = [2, 3, 2]  # part of the full dataset ([8, 9, 11])
    train_info, test_info, features_max_min, dataset_dir = load_dataset_info({"machines_per_group": machines_per_group, 'trainSplit':0.8}, need_resampling=True)
    print("train_info", train_info)
    print("test_info", test_info)
    print("features_max_min", features_max_min)
    print("max_min dict", features_max_min.to_dict())
    machines = get_machine_names(machines_per_group)
    print(machines)
    evset = load_omny_server_data_anomaly(machines=machines)
    print(evset)

    # plot the data
    # Plot the first 3 features
    evset.plot(indexes="machine-1-1", max_num_plots=3)

    # Plot the labels
    evset["label"].plot(indexes="machine-1-1")
    import matplotlib.pyplot as plt

    plt.show()

    # load a single machine
    single_machine = get_single_machine_name(1, 1)
    evset_single = load_omny_server_data_anomaly(machines=[single_machine])
    evset.plot(indexes="machine-1-1", max_num_plots=3)
    evset["label"].plot(indexes="machine-1-1")
    plt.show()

