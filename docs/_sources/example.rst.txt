.. highlight:: shell
.. rst-class:: justify

Examples and demonstrations
========================================

Please find here a list of commands that allows to start the demos. These demos suppose that your system has all the tools installed to run them.

As indicated in the README.md the most convenient approach is to rely on a container using Singularity or Apptainer
and use the receipe provided in the install folder. In this case, one expects the following container to exist: 

.. code-block:: console

    $ /path/to/containers/tf2_addons.2.11.0.sif

NOTES:
    - Once run, all demos produce logs in specific timestamped folders in the `experiments` folder.
    - Have a look at the logs reported on the Tensorboard tool using command

    .. code-block:: console

        $ apptainer exec /path/to/containers/tf2_addons.2.11.0.sif tensorboard 
        --logdir experiments/YOURTARGETSUBFOLDER


Classical centralised model optimisation
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- Regression problem, a simple case with :math:`y = ax^2 + b`.

.. code-block:: console

    $ apptainer run /path/to/containers/tf2_addons.2.11.0.sif -m deeplearningtools.experiments_manager  
    -u examples/regression/mysettings_curve_fitting.py

- Classification problem on cats and dogs

.. code-block:: console

    $ apptainer run /path/to/containers/tf2_addons.2.11.0.sif -m deeplearningtools.experiments_manager  
    -u examples/classification/mysettings_image_classification.py

- Timeseries forecasting with advanced csv files preprocessing

.. code-block:: console

    $ apptainer run /path/to/containers/tf2_addons.2.11.0.sif -m deeplearningtools.experiments_manager  
    -u examples/timeseries/mysettings_timeseries_forecasting.py

- Embedding timeseries with advanced csv files preprocessing

.. code-block:: console

    $ apptainer run /path/to/containers/tf2_addons.2.11.0.sif -m deeplearningtools.experiments_manager  
    -u examples/embedding/mysettings_1D_experiments.py


Decentralised federated model optimisation
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Running the previous regression problem example in a federated learning context.

- Simulation mode

Run everything on the same machine: parameter server and clients.

Client logs will appear in `/tmp/ray/session_last/runtime_resources/working_dir_files/`

.. code-block:: console
    
    $ apptainer run /path/to/containers/tf2_addons.2.11.0.sif -m deeplearningtools.start_federated_server 
    -sim --usersettings examples/federated/mysettings_curve_fitting.py 

- Distributed mode

Parameter server and clients run on different processes (and maybe machines).

1. Start server (and wait for ready state)

.. code-block:: console

    #On command panel 0:
    $ apptainer run /path/to/containers/tf2_addons.2.11.0.sif -m deeplearningtools.start_federated_server 
    --usersettings examples/federated/mysettings_curve_fitting.py 

2. Start each client into a specific command panel. Client logs will appear in experiments/examples/federated, one folder per client.

.. code-block:: console

    #On command panel 1:
    $ apptainer run /path/to/containers/tf2_addons.2.11.0.sif -m deeplearningtools.experiments_manager 
    --procID 1 -u examples/federated/mysettings_curve_fitting.py

    #On command panel 2:
    $ apptainer run /path/to/containers/tf2_addons.2.11.0.sif -m deeplearningtools.experiments_manager 
    --procID 2 -u examples/federated/mysettings_curve_fitting.py

    #On command panel 3:
    $ apptainer run /path/to/containers/tf2_addons.2.11.0.sif -m deeplearningtools.experiments_manager 
    --procID 3 -u examples/federated/mysettings_curve_fitting.py

    #On command panel 4:
    $ apptainer run /path/to/containers/tf2_addons.2.11.0.sif -m deeplearningtools.experiments_manager 
    --procID 4 -u examples/federated/mysettings_curve_fitting.py


Running tests
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

You may want to check regression issues using pytest. Some global test cases are defined in the `test_framework.py` script, run this as follows:

.. code-block:: console
    
    $ apptainer exec /path/to/containers/tf2_addons.2.11.0.sif pytest test_framework.py


Application case: Cats and dogs classification
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Reference: inspired from https://www.tensorflow.org/tutorials/images/classification


1. Training and validation
''''''''''''''''''''''''''''''''''

Start a train/val session using command (a singularity/apptainer container with an optimized version of Tensorflow is used here: `tf2_addons.2.11.0.sif`), see README.md to build it:

.. code-block:: console
    
    $ apptainer run --nv tf2_addons.2.11.0.sif.sif -m deeplearningtools.experiments_manager 
    --usersettings=examples/classification/mysettings_image_classification.py

Once done, check for and use in the following steps the resulting folder, say for example `/abs/path/to/deeplearningtools/experiments/examples/cats_dogs_classification/my_trials_learningRate0.001_nbEpoch15_dataAugmentFalse_dropout0.2_imgHeight150_imgWidth150_2023-04-03--22:05:36/`.


2. Serve the trained model
'''''''''''''''''''''''''''''''

Start a tensorflow model server on the produced experiment models using command (the -psi command permits to start tensorflow model server installed in a singularity/apptainer container, here tf_server.2.11.0.sif):

.. code-block:: console
    
    $ python3 -m deeplearningtools.start_model_serving 
    --model_dir /abs/path/to/deeplearningtools/experiments/examples/cats_dogs_classification/my_trials_learningRate0.001_nbEpoch15_dataAugmentFalse_dropout0.2_imgHeight150_imgWidth150_2023-04-03--22:05:36/ -psi /abs/path/to/tf2_addons.2.11.0.sif.sif 


3. Request the model
'''''''''''''''''''''''''

Start a client that sends continuous requests to the server making use of a connected webcam

.. code-block:: console

    $ apptainer run --nv tf2_addons.2.11.0.sif.sif -m deeplearningtools.experiments_manager 
    --predict_stream=-1 --model_dir /abs/path/to/deeplearningtools/experiments/examples/cats_dogs_classification/my_trials_learningRate0.001_nbEpoch15_dataAugmentFalse_dropout0.2_imgHeight150_imgWidth150_2023-04-03--22:05:36
