[![Build Status](https://app.travis-ci.com/albenoit/DeepLearningTools.svg?branch=master)](https://app.travis-ci.com/albenoit/DeepLearningTools)

# Train, monitor, evaluate and deploy/serve your Tensorflow Machine Learning models rapidly in a unified way !

Here is a set of python3 scripts that demonstrate the use of Tensorflow2.x for model optimization and deployment on your data (1D, 2D, 3D...).
The proposed toolchain enables different experiments (model training/validating) to be launched in a unified way. All models are automatically exported periodically to enable model deployment (serving in production).
All the resulting experiment logs can be compared. Model versioning is enabled. The hyperparameters management enables the Tensorboard HPARAMS interface.

This framework can be driven by higher-level tools such as [hyperopt](https://hyperopt.github.io/) to explore the hyperparameters space, etc. (see examples/hyperopt for demos)

@brief : the main script 'experiments_manager.py'  enables training, validating and serving Tensorflow2.x models with python3

@author : Alexandre Benoit, Professor at LISTIC lab, FRANCE

A quick presentation of the system is available [here](https://docs.google.com/presentation/d/1tFetD27PK9kt29rdwwZ6QKLYNDyJkHoCLT8GHweve_8/edit?usp=sharing), details are given below.

This work has been facilitated by intensive experiments conducted on the JeanZay French supercomputer (Grants 2020-AD011011418 and 2021-AD011011418R1) (http://www.idris.fr/jean-zay/)

## Main ideas put together:

* Training a model defined with tf.keras to manage training, validation and export in an easy and systematic way (no more forget your favourite methodology, callbacks from one experiment to the other).
* Using moving averages to store parameters with values smoothed along the last training steps to get more stable and more accurate served models.
* Automatic storage of all the model outputs on the validation dataset in order to observe some data projections on the TensorBoard for embedding understanding.
* Early stopping interrupting training if considered metrics (validation loss by default) exist and do not decrease over a long period.
* Make use of the tensorflow-serving-api to serve the model and dynamically load updated models, even while training is still running.
* A generic tensorflow-serving client codes to reuse the trained model on a single sample or streamed data.
* Each experiment is stored in a specific folder for model versioning and comparison.
* Restart training after failure made easy.
* Reproducible experiments with random _seeds
* Activate various optimization options such as XLA, mixed precision and multi-GPU processing
* Federated Learning (FedML), relying on the [Flower library](https://flower.dev/), few changes to switch from your classical training model to the federated version
* Kafka data pipeline management: users can produce data to a kafka pipeline and the model optimization can be fed by kafka pipelines ! *Have a look at install/kafka/README.md*. For a given and working centralized project, your experiment config file contains all information to produce data on other ressources.
* *Have a look at the examples folder to start from typical ML problem examples.*
* **News** : 
  * Federated Learning compliant
  * **XAI** (eXplainable Artificial Intelligence) compliant, check out our semantic segmentation XAI in *examples/xaie* [ICPR-XAIE2022 paper](https://hal.archives-ouvertes.fr/hal-03719597).

## Approach:

* A single script, experiments_manager.py, that manages all the train/val/export/serve process is provided to let you no more care about it.
* You write the experiment settings file that focuses on the experiment but avoid the machinery. You then define the expected variables and functions (datasets, learning rates, loss, etc.). This is enough work but only focused on the experiment.
* You write your model in a separate file following a basic function prototype. This will allow you to switch between models but still relying on the same experiment settings.
* You run the experiment and regularly look at the Tensorboard to monitor indicators, weight distributions, model output embedding, etc.
* You finally run the model relying on the Tensorflow serving API.

# Machine Setup (validated with tensorflow 2.9.x->2.11.0)

Recommended installation process is to rely on containers as shown below. Frozen Python package dependency list is reported in file requirements.txt and is used to build containers. Then you can also perform a classical but much less reproducible and stable standard Python (Anaconda) installation using that file too. 
Relying on Singularity or Apptainer containers allows you to build the machine as a single .sif file and reuse (copy/paste) it on any other machine (laptop, desktop, server) where Singularity or Apptainer is installed. This is a good way to keep your time avoiding multiple installation procedures, libraries conflict management and all this time-wasting stuff !
More information on Apptainer (open source fork of Singularity) : [Apptainer](https://apptainer.org/getting-started)

## Container based installation using [Apptainer](https://apptainer.org/getting-started) or [Singularity](https://sylabs.io/), RECOMMENDED:
Have a try with containers to get an off-the-shelf system ready to run on NVIDIA GPUs !
Apptainer and Singularity will build containers from (official) Tensorflow docker images. Choose between your preferred image from the [Tensorflow docker hub](https://hub.docker.com/r/tensorflow/tensorflow/tags/) or from [NVIDIA NGC](https://www.nvidia.com/en-us/gpu-cloud/containers/).

I consider here Singularity or the open source fork Apptainer very close to Docker but generally more adopted for HPC. However an equivalent container design can be done using Docker!
### Notes on Singularity/Apptainer:
#### install (as root) :
  * [Singularity](https://sylabs.io/docs/)
  * [Apptainer (open sourced fork)](https://apptainer.org/docs)
#### build the image with GPU (as root):
  * build a custom image with the provided *install/tf2_addons.def* file that includes all python packages to build the container :
  * the install/tf_server.def file is also provided to build a Tensorflow model server container.
```
sudo apptainer build tf2_addons.sif tf2_addons.def #container for model training and validation
sudo apptainer build tf_server.sif tf_server.def               #container for model serving only
```
### run the image (as standard user):
  * open a shell on this container, bind to your system folders of interest : `apptainer shell --nv --bind /path/to/your/DeepLearningTools/:DeepLearningTools/ tf2_addons.sif`
  * run the framework, for example on the curve fitting example: `cd /DeepLearningTools/` followed by `python experiments_manager.py --usersettings examples/regression/mysettings_curve_fitting.py`
  * if the gpu is not found (error such as `libcuda reported version is: Invalid argument: expected %d.%d, %d.%d.%d, or %d.%d.%d.%d form for driver version; got "1"`, sometimes, NVIDIA module should be reloaded after a suspend period. Recover it using command `nvidia-modprobe -u -c=0`

## Manual installation using Anaconda and pip (NOT RECOMMENDED, NOT MAINTAINED).
The requirements.txt python package list is still updated. However, some other system tools are also required such that building the software environment as a container is the recommended approach presented above. However, you can still use this requirements file to build up classical python virtual environment but this is not recommended nor maintained. In this case, you then have to check all non python additional tools listed in the *install/tf2_addons.def* file.

### Anaconda installation (local account installation, non-root installations, recommended):
1. download and install the appropriate anaconda version from here: https://www.anaconda.com/distribution/
2. create a specific environment to limit interactions with the system installation:
conda create --name tf_gpu
sometimes required: source ~/install/anaconda3/etc/profile.d/conda.sh
3. activate your environment before installing the packages and run your python scripts:
conda activate tf_gpu
3. install the set of required packages (opencv, gdal and scikit-learn are not required for all scripts):
conda install tensorflow-gpu pandas opencv matplotlib gdal gdal scikit-learn
tensorflow_serving api is available elsewhere from this command:
conda install -c qiqiao tensorflow_serving_api

### pip installation (local system installation install as root):
1. install python 3.x and the associated python pip, maybe create a specific environment with the virtualenv tool.
2. install Tensorflow, Tensorflow serving and related tools using the install/requirements.txt file. It includes those packages and associated tools (opencv, pandas, etc.) : pip install -r requirements.txt

# How to train/test/serve a model ?

The main script is experiments_manager.py can be used in 3 modes, here are some command examples:
## 1. Train a model in a context specified in a parameter script such as examples/regression/mysettings_curve_fitting.py (details provided in the following TODO section):
 * ***RECOMMENDED :if all the libraries are installed in a singularity container located at /path/to/tf2_addons.sif***
```
apptainer run --nv /path/to/tf2_addons.sif experiments_manager.py --usersettings examples/regression/mysettings_curve_fitting.py
```

 * ***if all the libraries are system installed***
```
python experiments_manager.py --usersettings=examples/regression/mysettings_curve_fitting.py
```
## 2.start a Tensorflow server on the trained/training model :

 * ***RECOMMENDED : if tensorflow_model_server is installed on a singularity container located at /path/to/tf_server.sif***
   relying on a lightweight host installation (python3 and standard libs, no more requirements)
```
python3 start_model_serving.py --model_dir=/absolute/path/to/experiments/example/curve_fitting/my_test_2018-01-03--14:40:53 -psi=/absolute/path/to/tf_server.sif
```

 * ***if tensorflow_model_server is installed on the system as well as the python libraries***
```
python experiments_manager.py --start_server --model_dir=experiments/examples/curve_fitting/my_test_2018-01-03--14:40:53
```
   or 
```
python3 start_model_serving.py --model_dir=experiments/curve_fitting/my_test_2018-01-03--14:40:53
```

## 3. Request Tensorflow model server, sending input buffers and receiving answers

 * ***RECOMMENDED : if all the libraries are installed in a singularity container located at /path/to/tf2_addons.sif***
```
apptainer run --nv /path/to/tf2_addons.sif experiments_manager.py --predict_stream=-1 --model_dir=experiments/curve_fitting/my_test_2018-01-03--14\:40\:53/
```

 * ***if all the libraries are system installed***
```
python experiments_manager.py --predict_stream=-1 --model_dir=experiments/curve_fitting/my_test_2018-01-03--14\:40\:53/
```

## NOTE :

once trained (or along training), start the Tensorboard parsing logs of
the experiments folder (provided example is experiments/1Dsignals_clustering):
from the scripts directory using command:
```
tensorboard  --logdir=experiments/curve_fitting
```
Then, open a web browser and reach http://127.0.0.1:6006/ to monitor training
values and observe the obtained embedding.

# DESIGN:

1. The main code for training, validation and prediction is specified in the main script (experiments_manager.py).
2. Most of the use case specific parameters and Input/Output functions have been
moved to a separated setting script such as 'examples/regression/mysettings_curve_fitting.py' that is targeted when starting the script (this
  filename is set in var FLAGS.usersettings in the main script).
3. The model to be trained and served is specified in a different script targeted by the settings file.

# Notes on Federated Learning
The framework is designed to switch with few changes from the classical centralized version to federated learning. First create and validate a classical centralized experiment and model. Then, relying on few changes, switch and compare with a Federated Learning relying on the [Flower library](https://flower.dev/). More into the details, have a look at the specific *examples/federated/README.md* file. As a brief summary, here are some basics showing you how to adjust your experiment settings file to get something like *examples/federated/mysettings_curve_fitting.py*:

  * Add ```hparams['federated']='fedavg'``` to specify the key 'federated' and the chosen aggregation method.
  * Add ```hparam['procID']=X``` with *X* an integer value to let this experiment run as a federated client (with process ID *X* in this example).
  * Adjust your data sources in the settings file in order to be able to switch from one data surce to another simply relying on the ```hparam['procID']=X``` parameter.
  * Start the federated server with the *start_federated_server.py* python script and go !
  ```
  apptainer run install/tf2_addons.2.9.1.sif start_federated_server.py --usersettings examples/federated/mysettings_curve_fitting.py
  ```
  * Start clients from the command line and update their process ID with the *--procID* option, for instance  *--procID 20*, that will automatically update ```hparam['procID']=20``` for *this* client: 
  
  ```
  apptainer run  install/tf2_addons.2.9.1.sif experiments_manager.py --procID 20  --usersettings examples/federated/mysettings_curve_fitting.py 
  ```
# Notes on Kafka
Data preprocessing can sometimes be an intensive task that may compete with the training steps such that it should be decentralized. To do so, starting from a working centralized experiment configuration, as for the Federated learning option, Kafka data pipelining should be enabled with minimal changes. The main principle is to decentralise data production and feed a kafka log queue that will be read by a processing node dedicated to the model optimization part, no more managing data preprocessing. 
The general idea is then to apply few changes on a classical centralized experiment settings file and reuse it as is on the data production nodes and on the model optimization node. Detailed informations are provided in *install/kafka/REAME.md* but here is a brief summary:
 * suppose that a Kafka server is available somewhere. For a fresh config, maybe check *install/kafka* that provides scripts to build a kafka node on a docker container.
 * On the data production nodes, simply start production process relying on the experiment settings file, that specifies, as you currently do, parameters and more specifically, the *get_input_pipeline* function. Then, start data production using command:
 ```
 apptainer run --nv install/tf2_addons.2.8.0.sif start_kafka_producer.py --server localhost:9092 --procID 0 --usersettings examples/regression/mysettings_curve_fitting.py
 ```
 with prodID, as for the federated learning discussion above, an integer that could help switch from one dataset to another one.

 * On the node that conducts model optimization, simply specify that data is available on a kafka log queue by adding to the experiment settings file:
   * Variable `consume_data_from_kafka=True`
   * Specify server(s) IP (adjust the following line as required) with variable `kafka_bootstrap_servers=['localhost:9092']`

# KNOWN ISSUES :

* Scripts is now only dedicated to Tensorflow 2. Last Tensorflow 1.14 supports is tagged v1 in the repository and has been much simplified after the tf2 migration.
* This framework is intended to help design, optimize and deploy Tensorflow based models with some systematic strategy that stabilizes the workflow and user experience. However, this is built for research and is subject to changes and updates are impacted by the maintainer activities.

# TODO :

To adapt to new case studies, just update the closest example (examples folder) experiment file mysettingsxxx.py and adjust I/O functions.
For any experiment, the availability of all the required fields in the settings file is checked by the tools/experiments_settings.py script. Exceptions are raised on errors, have a look in this file to ensure you prepared everything right and compare your settings file to the provided examples.

