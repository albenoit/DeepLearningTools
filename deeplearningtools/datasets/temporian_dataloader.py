""" Temporian dataloader for TensorFlow datasets

@author: Alexandre Benoit, LISTIC lab, FRANCE
@brief : a set of helpers to load data using the Temporian library and feed Tensorflow datasets
"""

from typing import List

import temporian as tp
import tensorflow as tf


def temporian_load_csv(
    filename: str,
    timestamps_colname: str,
    sep: str = ";",
    selected_features: List = None,
    raw_data_cast: dict = None,
):
    """Create a Temporian even set from a CSV file

    Args:
            filename (str): path to the target file
            timestamps_colname (str): the name of the column that represents the timestamps
            sep (str): column separator
            selected_features (List): optionnal, the list of column names to consider, all others would be discarded
            raw_data_preproc (dict): optionnal, the dictionnary with keys 'cast', 'scale', 'offset', 'fillna'
                (and maybe more later) that specify transformations to be applied for each feature,
                 specifying, each time an inner dict of 'featurename':param to apply just after data is loaded

    Returns:
            a Temporian eventset
    """
    # Load the data
    raw_data = tp.from_csv(
        filename,
        timestamps=timestamps_colname,
        sep=sep,
    )
    @tp.compile
    def load_tp_data(tp_events: tp.EventSet):
        if raw_data_cast is not None:# immediately cast the data if required
            if 'cast' in raw_data_cast.keys():
                tp_events = tp_events.cast(raw_data_cast['cast'])

            if False:#'offset' in raw_data_cast.keys() and 'scale' in raw_data_cast.keys():
                def normalize(x):
                    for feature in selected_features:
                        x[feature] = (x[feature] - raw_data_cast['offset'][feature]) / raw_data_cast['scale'][feature]
                    return x
                tp_events = tp_events.map(normalize)
                
        feature_names = tp_events.schema.feature_names()
        print("feature_names", feature_names)
        data = tp_events
        if selected_features is not None:
            if len(selected_features) > 0:
                data = tp_events[selected_features]

            # select columns of interest AND add sample enumeration
            data = tp.glue(data.enumerate(), data)
            return data
    data=load_tp_data(raw_data)
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


def temporian_to_tfdataset(
    temporian_data: List[tp.EventSet],
    temporal_series_length: int,
    batch_size: int=0,
    prepare_example_fn: callable = None,
    shuffle_size: int = 100,
    filter_fn: callable = None,
    jitter: bool = False,
):
    """
    Returns a TensorFlow dataset for training or evaluation from a Temporian evenset.

    Args:
            temporian_data (tp.EventSet): a temporian eventset.
            temporal_series_length (int): The length of each temporal series.
            batch_size (int): the batch size. If 0, the no batching is applied.
            filter_fn (fl): an optional function to filter the dataset BEFORE prepare_example_fn is applied.
            prepare_example_fn (fn): a function that receives a single data sample and deduces related (input,label)
            shuffle_size (int): if >0, then shuffle the related last values.
            jitter (bool, optional): Whether to apply jittering on the temporal axis. Defaults to False.

    Returns:
            tf.data.Dataset: The TensorFlow dataset.

    """

    if jitter:
        n_events_orig = len(temporian_data.get_index_value(()))
        if "enumerate" not in temporian_data.schema.feature_names():
            sample_positions = temporian_data.enumerate()
        else:
            sample_positions = temporian_data["enumerate"]

        sample_offset = (
            tf.random.uniform(
                shape=[], minval=0, maxval=temporal_series_length, dtype=tf.int64
            )
            .numpy()
            .item()
        )
        temporian_data = temporian_data.filter(sample_positions > sample_offset)
        n_events_after = len(temporian_data.get_index_value(()))
        print("n events (before, after) jitter", (n_events_orig, n_events_after))

    ts_dataset = tp.to_tensorflow_dataset(
        temporian_data
    ).prefetch(tf.data.AUTOTUNE)  # convert to tensorflow dataset

    ts_dataset_seq = ts_dataset.batch(
        temporal_series_length, drop_remainder=True
    )  # group time steps into sequences

    if filter_fn is not None:  # remove some samples
        ts_dataset_seq = ts_dataset_seq.filter(filter_fn)

    if prepare_example_fn is not None:
        ts_dataset = ts_dataset_seq.map(
            prepare_example_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=shuffle_size <= 0,# if not shuffling as for validation set, ensure determinism
        )  # extract data and labels of interest

    # finally prepare time series batchs
    if shuffle_size>0:
        ts_dataset = ts_dataset.shuffle(shuffle_size)
    if batch_size >0:
        ts_dataset = ts_dataset.batch(batch_size)

    return ts_dataset.prefetch(tf.data.AUTOTUNE)

# global function that chains the csv loading and the tfdataset creation
def build_timeseries_tfdataset(
    filename: str,
    timestamps_colname: str,
    sep: str,
    temporal_series_length: int,
    batch_size: int=0,
    selected_features: List = None,
    raw_data_preproc:dict = None,
    prepare_example_fn: callable = None,
    shuffle_size: int = 100,
    filter_fn: callable = None,
    jitter: bool = False,
):
    """
    Load a temporian dataset from a CSV file and convert it to a TensorFlow dataset.

    Args:
            filename (str): path to the target file
            timestamps_colname (str): the name of the column that represents the timestamps
            sep (str): column separator
            selected_features (List): optionnal, the list of column names to consider, all others would be discarded
            raw_data_preproc (dict): optionnal, the dictionnary with keys 'cast', 'scale', 'offset', 'fillna'
                (and maybe more later) that specify transformations to be applied for each feature,
                 specifying, each time an inner dict of 'featurename':param to apply just after data is loaded
            temporal_series_length (int): The length of each temporal series.
            batch_size (int): the batch size. If 0, the no batching is applied.
            filter_fn (fl): an optional function to filter the dataset BEFORE prepare_example_fn is applied.
            prepare_example_fn (fn): a function that receives a single data sample and deduces related (input,label)
            shuffle_size (int): if >0, then shuffle the related last values.
            jitter (bool, optional): Whether to apply jittering on the temporal axis. Defaults to False.

    Returns:
            tf.data.Dataset: The TensorFlow dataset.
    """
    data = temporian_load_csv(
        filename=filename,
        timestamps_colname=timestamps_colname,
        sep=sep,
        selected_features=selected_features,
        raw_data_cast=raw_data_preproc,

    )

    return temporian_to_tfdataset(
        temporian_data=data,
        temporal_series_length=temporal_series_length,
        batch_size=batch_size,
        prepare_example_fn=prepare_example_fn,
        shuffle_size=shuffle_size,
        filter_fn=filter_fn,
        jitter=jitter,
    )


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


def test_build_timeseries_tfdataset():
    selected_features = [
        "RDC-ChambreEnfants_CO2_GAS_CONCENTRATIONppm<>Avg",
        "RDC-ChambreParents_CO2_GAS_CONCENTRATIONppm<>Avg",
        "RDC-Séjour_CO2_GAS_CONCENTRATIONppm<>Avg",
    ]
    selected_labels = ["RDC-Séjour_Humidité_HUMIDITY%<>Avg"]

    temporal_series_length = 20
    batch_size = 5

    def extract_label(example):
        data_cols = {
            feat: tf.convert_to_tensor(example.pop(feat)) for feat in selected_features
        }
        label_cols = {
            feat: tf.convert_to_tensor(example.pop(feat)) for feat in selected_labels
        }
        return data_cols, label_cols

    dataset = build_timeseries_tfdataset(
        filename="datasamples/timeseries/House1_bid-data_127.csv",
        timestamps_colname="firstDate",
        sep=",",
        selected_features=selected_features + selected_labels,
        temporal_series_length=temporal_series_length,
        batch_size=batch_size,
        prepare_example_fn=extract_label,
        shuffle_size=10,
        # filter_fn=filter_fn,
        jitter=True,
    )

    for data, labels in dataset.take(1):
        assert data is not None
        assert labels is not None


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # global configuration hyperparameters
    selected_features = [
        "RDC-ChambreEnfants_CO2_GAS_CONCENTRATIONppm<>Avg",
        # "RDC-ChambreParents_CO2_GAS_CONCENTRATIONppm<>Avg",
        # "RDC-Séjour_CO2_GAS_CONCENTRATIONppm<>Avg",
    ]
    selected_labels = ["RDC-Séjour_Humidité_HUMIDITY%<>Avg"]

    temporal_series_length = 5
    batch_size = 1

    """ FIRST APPROACH, more control by chaining the functions and allows to split the data
     before creating the tensorflow dataset
    """
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
    # define a local filter function (optionnal, here we remove non finite values)
    def filter_is_finite(example):
        return tf.reduce_all(
            [tf.math.is_finite(example[key]) for key in selected_features+selected_labels]
        )  # data_is_finite and label_is_finite

    train_dataset = temporian_to_tfdataset(
        temporian_data=train_data,
        temporal_series_length=temporal_series_length,
        batch_size=batch_size,
        prepare_example_fn=extract_label,
        shuffle_size=10,
        filter_fn=filter_is_finite,
        jitter=True,
    )
    val_dataset = temporian_to_tfdataset(
        temporian_data=val_data,
        temporal_series_length=temporal_series_length,
        batch_size=batch_size,
        prepare_example_fn=extract_label,
        shuffle_size=0,
        filter_fn=filter_is_finite,
        jitter=False,
    )
    """
    for data, labels in train_dataset.take(1):
        print("data", data)
        print("labels", labels)
        plt.plot(data["RDC-ChambreEnfants_CO2_GAS_CONCENTRATIONppm<>Avg"][0])
    """

    """ SECOND APPROACH, all in one function, more appropriate for a single dataset
    """
    # all in one function
    dataset = build_timeseries_tfdataset(
        filename="datasamples/timeseries/House1_bid-data_127.csv",
        timestamps_colname="firstDate",
        sep=",",
        selected_features=selected_features + selected_labels,
        temporal_series_length=temporal_series_length,
        batch_size=batch_size,
        prepare_example_fn=extract_label,
        shuffle_size=10,
        filter_fn=filter_is_finite,
        jitter=True,
    )

    for data, labels in dataset.take(1):
        print("data", data)
        print("labels", labels)
        plt.plot(data["RDC-ChambreEnfants_CO2_GAS_CONCENTRATIONppm<>Avg"][0])
        plt.show()
    print("done")
