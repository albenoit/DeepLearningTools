# Example of Federated Learning
Simple integration of federated learning into the framework.
Federated Learning is enabled by the Flower library (https://flower.dev/docs/quickstart_tensorflow.html).

Example based on curve regression demo (very similar to examples/regression demo). Changes to move to fedml:
-add variable to the settings script: enable_federated_learning=True
-create different settings files, one for each client, here, only the sample data ranges vary between the 2 clients: each one learns a different par of the curve

# How to launch demo
## First start the centralized parameter server:
rely on the start_federated_server.py placed at the root folder to start the server:

### train from scrath:
singularity run /path/to/singularity/container/tf2_addons.sif start_federated_server.py --usersettings examples/federated/mysettings_curve_fitting.py 

### train from a pretrained model (meta-learning):
-> if you need to start server with a pretrained model, add the -w 'path/to/previous/experiment' command, for instance:

 singularity run  /path/to/singularity/container/tf2_addons.sif start_federated_server.py  --usersettings examples/federated/mysettings_curve_fitting.py -w experiments/examples/curve_fitting/my_test_hiddenNeurons50_predictSmoothParamsTrue_learningRate0.1_nbEpoch5000_addNoiseTrue_anomalyAtX-3_procID0_2022-01-21--21\:30\:03/

## Second start different learning clients
In this example, we rely on the same settings script but specify a process id (--procID x). This procID is used in the settings file to adjust the input data pipeline, each client has a different input data setup

 singularity run  /path/to/singularity/container/tf2_addons.sif experiments_manager.py --usersettings examples/federated/mysettings_curve_fitting.py --procID 1

 singularity run  /path/to/singularity/container/tf2_addons.sif experiments_manager.py --usersettings examples/federated/mysettings_curve_fitting.py --procID 2

 singularity run  /path/to/singularity/container/tf2_addons.sif experiments_manager.py --usersettings examples/federated/mysettings_curve_fitting.py --procID 3

# Adding your own federated learning ideas

You may want to add and test a new strategy or client behavior. To do so, look at python files in deeplearningtools/helpers/federated:
 . update the client behaviors in flclient.py
 . create a new strategy following template_strategy.py (follow the official Flwr guidelines)
