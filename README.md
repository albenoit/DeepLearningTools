# Train, monitor, evaluate and deploy/serve your Tensorflow Machine Learning models rapidly in a unified way !

Here is a set of python3 scripts that demonstrate the use of Tensorflow2.x for model optimization and deployment on your data (1D, 2D, 3D...).
The proposed tool-chain enables different experiments (model training/validating) to be launched in a unified way. All models are automatically exported periodically to enable model deployment (serving in production).
All the resulting experiments logs can be compared. Model versioning is enabled. Hyperparameters management enables the Tensorboard HPARAMS interface.

This framework can be driven by higher level tools such as [hyperopt](https://hyperopt.github.io/) to explore the hyperparameters space, etc. (see examples/hyperopt for demo(s))

@brief : the main script 'experiments_manager.py' enables training, validating and serving Tensorflow2.x models with python3

@author : Alexandre Benoit, LISTIC lab, FRANCE

A quick presentation of the system is available [here](https://docs.google.com/presentation/d/1tFetD27PK9kt29rdwwZ6QKLYNDyJkHoCLT8GHweve_8/edit?usp=sharing), details are given below.

## Main ideas put together:

* Training a model defined with tf.keras to manage training, validation and export in a easy and systematic way (no more forget your favorite methodology, callbacks from one experiment to the other).
* Using moving averages to store parameters with values smoothed along the last training steps to get more stable and more accurate served models.
* Automatic storage of all the model outputs on the validation dataset in order to observe some data projections on the TensorBoard for embedding understanding.
* Early stopping to interrupt training if the considered metric (validation loss by default) exist and does not decrease over a long period.
* Make use of the tensorflow-serving-api to serve the model and dynamically load updated models, even while training is still running.
* A generic tensorflow-serving client codes to reuse the trained model on single sample or streamed data.
* Each experiment is stored in a specific folder for model versioning and comparison.
* Restart training after failure made easy.
* Reproducible experiments with random_seeds
* *Have a look at the examples folder to start from typical ML problem examples.*

## Approach:

* A single script, experiments_manager.py, that manages all the train/val/export/serve process is provided to let you no more care about it.
* You write the experiment settings file that focuses on the experiment but avoid the machinery. You then define the expected variables and functions (datasets, learning rates, loss, etc.). This is enough work but only focused on the experiment.
* You write your model in a separate file following a basic function prototype. This will allow you to switch between models but still relying on the same experiment settings.
* You run the experiment and regularly look at the Tensorboard to monitor indicators, weight distributions, model output embedding, etc.
* You finally run the model relying on the Tensorflow serving API.

# Machine Setup (validated with tensorflow 2.0 and 2.1)

Python package installation can be managed relying on Anaconda or pip package managers. You can install the required packages manually, as shown below. However, the most convenient way is to consider containers in order to keep your system as is (safe and stable) and install all the required packages (maybe in different versions) apart without conflicts. In addition, the same built container can be deployed on laptops, desktops and servers (in clouds) by simply copy/paste of the built container on the target machine. Keep your time avoiding multiple installation procedures, libraries conflict management and all this time-wasting stuff !
More information on the interest of Singularity :
*  a brief summary : https://www.nextplatform.com/2017/04/10/singularity-containers-hpc-reproducibility-mobility/
*  talks about Singularity : https://sylabs.io/videos

## Container based installation using Singularity (https://sylabs.io/), recommended:
Have a try with containers to get an off-the-shelf system ready to run on NVIDIA GPUs !
Singularity will build containers from (official) Tensorflow docker images. Choose between your preferred image from the Tensorflow docker hub https://hub.docker.com/r/tensorflow/tensorflow/tags/ or from NVIDIA NGC https://www.nvidia.com/en-us/gpu-cloud/containers/ .

I consider here Singularity instead of Docker for some reason such as use simplicity, reduced image size, HPC deployment capability, checkout there : https://sylabs.io/ . However an equivalent container design can be done using Docker!
## Notes on singularity:
### install singularity (as root) :
  * debian installation : https://wiki.debian.org/singularity
### build the image with GPU (as root):
  * build a custom image with the provided *tf2_addons.def* file that includes all python packages to build the container :
  * the tf_server.def file is also provided to build a tensorflow model server container.
```
singularity build tf2_addons.sif tf2_addons.def #container for model training and validation
singularity build tf_server.sif tf_server.def               #container for model serving only
```
### run the image (as standard user):
  * open a shell on this container, bind to your system folders of interest : `singularity shell --nv --bind /path/to/your/DeepLearningTools/:DeepLearningTools/ tf_nv_addons.sif`
  * run the framework, for example on the curve fitting example: `cd /DeepLearningTools/` followed by `python experiments_manager.py --usersettings examples/regression/mysettings_curve_fitting.py`
  * if the gpu is not found (error such as `libcuda reported version is: Invalid argument: expected %d.%d, %d.%d.%d, or %d.%d.%d.%d form for driver version; got "1"`, sometimes, NVIDIA module should be reloaded after a suspend period. Recover it using command `nvidia-modprobe -u -c=0`

## Manual installation using Anaconda and pip (WARNING, package list is no more updated, have a look at the container definition *.def files to see the updated required packages list)
### anaconda installation (local account installation, non root installation, recommended):
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

### pip installation (local system installation, install as root):
1. install python 2.7 or 3.x and the associated python pip, maybe create a specific environment with the virtualenv tool.
2. install Tensorflow, Tensorflow serving and related tools using the requirements.txt file. It includes those packages and associated tools (opencv, pandas, etc.) : pip install -r requirements.txt

# How to train/test/serve a model ?

The main script is experiments_manager.py can be used in 3 modes, here are some command examples:
1. train a model in a context specified in a parameter script such as examples/regression/mysettings_curve_fitting.py (details provided in the following TODO section):

  * if all the libraries are system installed
```
python experiments_manager.py --usersettings=examples/regression/mysettings_curve_fitting.py
```
  * if all the libraries are installed in a singularity container located at **/path/to/tf2_addons.sif**
```
singularity run --nv /path/to/tf2_addons.sif experiments_manager.py --usersettings examples/regression/mysettings_curve_fitting.py
```
2. start a Tensorflow server on the trained/training model :
  * if tensorflow_model_server is installed on the system
```
python experiments_manager.py --start_server --model_dir=experiments/curve_fitting/my_test_2018-01-03--14:40:53
```
  * if tensorflow_model_server is installed on a singularity container located at **/path/to/tf_server.sif**
```
python experiments_manager.py --start_server --model_dir=experiments/curve_fitting/my_test_2018-01-03--14:40:53 -psi=/path/to/tf_server.sif
```
3. interact with the Tensorflow server, sending input buffers and receiving answers,
```
python experiments_manager.py --predict --model_dir=experiments/curve_fitting/my_test_2018-01-03--14\:40\:53/
```

## NOTE :

once trained (or along training), start the Tensorboard to parse logs of
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
moved to a separated settings script such as 'examples/regression/mysettings_curve_fitting.py' that is targeted when starting the script (this
  filename is set in var FLAGS.usersettings in the main script).
3. The model to be trained and served is specified in a different script targeted by the settings file.

# KNOWN ISSUES :

*  Scripts is now only dedicated to Tensorflow 2. Last Tensorflow 1.14 support is tagged v1 in the repository and has been much simplified after the tf2 migration.
*  Migration to Tensorflow 2 is still on the way, some stuff needs to be recovered : training restart after interruption, embeddings projection display on Tensorboard, work in progress, contributions are welcome !

# TODO :

To adapt to new case studies, just update the mysettingsxxx.py file and adjust I/O functions.
For any experiment, the availability of all the required fields in the settings file is checked by the tools/experiments_settings.py script. Exceptions are raised on errors, have a look in this file to ensure you prepared everything right and compare your settings file to the provided examples.
In addition and as a reminder, here are the functions prototypes:

