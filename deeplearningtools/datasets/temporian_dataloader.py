""" Temporian dataloader for TensorFlow datasets

@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : a set of helpers to load data using the Temporian library and feed Tensorflow datasets
"""

from typing import List
import temporian as tp
import tensorflow as tf

def temporian_load_csv(
    filename: str, timestamps_colname: str, sep: str = ";", selected_features: List = []
):
    """Create a Temporian even set from a CSV file

    Args:
            filename (str): path to the target file
            timestamps_colname (str): the name of the column that represents the timestamps
            sep (str): column separator
            selected_features (List): optionnal, the list of column names to consider, all others would be discarded

    Returns:
            a Temporian eventset
    """
    # Load the data
    raw_data = tp.from_csv(
        filename,
        timestamps=timestamps_colname,
        sep=sep,
    )

    feature_names = raw_data.schema.feature_names()
    print("feature_names", feature_names)
    if len(selected_features) > 0:
        data = raw_data[selected_features]
    else:
        data = raw_data
    # select columns of interest AND add sample enumeration
    data = tp.glue(raw_data.enumerate(), data)
    n_events = len(data.get_index_value(()))
    print("Number of events", n_events)
    return data


def temporian_train_val_test_split(
    data: tp.EventSet, train_ratio: float = 0.6, val_ratio: float = 0.2
):
    """
    Splits the given data into training, validation, and testing sets based on the provided ratios.

    Args:
            data (tp.EventSet): The input temporian data to be split.
            train_ratio (float, optional): The ratio of data to be used for training. Defaults to 0.6.
            val_ratio (float, optional): The ratio of data to be used for validation. Defaults to 0.2.

    Returns:
            tuple: A tuple containing the training, validation, and testing sets.
    """

    if train_ratio + val_ratio > 1:
        raise ValueError("Train and val split ratio >1")
    n_events = len(data.get_index_value(()))
    train_until = int(n_events * train_ratio)
    val_until = train_until + int(n_events * val_ratio)

    # split partitions
    if "enumerate" not in data.schema.feature_names():
        data = tp.glue(data.enumerate(), data)
    sample_positions = data["enumerate"]
    train_data = data.filter(sample_positions <= train_until)
    val_data = data.filter(
        (sample_positions > train_until) & (sample_positions <= val_until)
    )
    test_data = data.filter(sample_positions > val_until)

    return train_data, val_data, test_data


def get_tf_dataset(
    temporian_data,
    temporal_series_length: int,
    batch_size: int,
    shuffle: bool,
    jitter: bool = False,
):
    """
    Returns a TensorFlow dataset for training or evaluation from a Temporian evenset.

    Args:
            temporian_data (tp.EventSet): The input data.
            temporal_series_length (int): The length of each temporal series.
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the dataset.
            jitter (bool, optional): Whether to apply jittering. Defaults to False.

    Returns:
            tf.data.Dataset: The TensorFlow dataset.

    """

    if jitter:
        n_events_orig = len(temporian_data.get_index_value(()))
        if "enumerate" not in data.schema.feature_names():
            sample_positions = data.enumerate()
        else:
            sample_positions = temporian_data["enumerate"]

        sample_offset = tf.random.uniform(
            shape=[], minval=0, maxval=temporal_series_length, dtype=tf.int64
        ).numpy().item()
        print("sample_offset", sample_offset, type(sample_offset))
        temporian_data = temporian_data.filter(sample_positions > sample_offset)
        n_events_after = len(temporian_data.get_index_value(()))
        print("n events (before, after) jitter", (n_events_orig, n_events_after))

    ts_dataset = tp.to_tensorflow_dataset(
        temporian_data
    )  # convert to tensorflow dataset

    ts_dataset_seq = ts_dataset.batch(
        temporal_series_length, drop_remainder=True
    )  # group time steps into sequences
    ts_dataset = ts_dataset_seq.map(
        extract_label
    )  # extract data and labels of interest
    # finally prepare time series batchs
    if shuffle:
        ts_dataset = ts_dataset.shuffle(batch_size * 100)
    dataset = ts_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    """print('ts_dataset', ts_dataset, type(ts_dataset))
                print('ts_dataset_seq', ts_dataset_seq)
                print('one raw sample     =', ts_dataset.take(1))
                print('one sample sequence=', ts_dataset_seq.take(1))
                print('dataset', dataset)
                """
    return dataset


def test_temporian_load_csv():
    selected_features = [
        "RDC-ChambreEnfants_CO2_GAS_CONCENTRATIONppm<>Avg",
        "RDC-ChambreParents_CO2_GAS_CONCENTRATIONppm<>Avg",
        "RDC-Séjour_CO2_GAS_CONCENTRATIONppm<>Avg",
    ]

    data = temporian_load_csv(
        filename="datasamples/timeseries/House1_bid-data_127.csv",
        timestamps_colname="firstDate",
        sep=",",
        selected_features=selected_features,
    )

    n_events = len(data.get_index_value(()))
    assert n_events == 1000


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # global configuration hyperparameters
    selected_features = [
        "RDC-ChambreEnfants_CO2_GAS_CONCENTRATIONppm<>Avg",
        "RDC-ChambreParents_CO2_GAS_CONCENTRATIONppm<>Avg",
        "RDC-Séjour_CO2_GAS_CONCENTRATIONppm<>Avg",
    ]
    selected_labels = ["RDC-Séjour_Humidité_HUMIDITY%<>Avg"]
    
    temporal_series_length=20
    batch_size=5

    # load data    
    data = temporian_load_csv(
        filename="datasamples/timeseries/House1_bid-data_127.csv",
        timestamps_colname="firstDate",
        sep=",",
        selected_features=selected_features + selected_labels,
    )

    n_events = len(data.get_index_value(()))
    print("Number of events", n_events)
    data.plot()

    train_data, val_data, test_data = temporian_train_val_test_split(data, 0.6, 0.2)
    train_data.plot()
    val_data.plot()
    test_data.plot()
    plt.show()

    # a local method that describes how to extract data columns and labels
    def extract_label(example):
        print("example itemps", example)
        data_cols = {
            feat: tf.convert_to_tensor(example.pop(feat)) for feat in selected_features
        }
        label_cols = {
            feat: tf.convert_to_tensor(example.pop(feat)) for feat in selected_labels
        }
        return data_cols, label_cols

    train_dataset = get_tf_dataset(
        temporian_data=train_data,
        temporal_series_length=temporal_series_length,
        batch_size=batch_size,
        shuffle=True,
        jitter=True,
    )
    val_dataset = get_tf_dataset(
        temporian_data=val_data,
        temporal_series_length=temporal_series_length,
        batch_size=batch_size,
        shuffle=False,
        jitter=False,
    )

    for data, labels in train_dataset.take(1):
        print("data", data)
        print("labels", labels)
        plt.plot(data["RDC-ChambreEnfants_CO2_GAS_CONCENTRATIONppm<>Avg"][0])
        plt.show()

