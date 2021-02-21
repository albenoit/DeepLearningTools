# Example of Federated Learning
Simple integration of federated learning into the framework.
Federated Learning is enabled by the Flower library (https://flower.dev/docs/quickstart_tensorflow.html).

Example based on curve regression demo (very similar to examples/regression demo). Changes to move to fedml:
-add variable to the settings script: enable_federated_learning=True
-create different settings files, one for each client, here, only the sample data ranges vary between the 2 clients: each one learns a different par of the curve

# How to launch demo
## First start the centralized parameter server:
singularity run /home/alben/install/containers/tf2_addons.2.4.1.jupyter.fl.sif examples/federated/start_server.py

## Second start different learning clients
In this example, we rely on the same settings script but specify a process id (--procID x). This procID is used in the settings file to adjust the input data pipeline, each client has a different input data setup


singularity run /home/alben/install/containers/tf2_addons.2.4.1.jupyter.fl.sif experiments_manager.py --usersettings examples/federated/mysettings_curve_fitting.py --procID 1

singularity run /home/alben/install/containers/tf2_addons.2.4.1.jupyter.fl.sif experiments_manager.py --usersettings examples/federated/mysettings_curve_fitting.py --procID 2

singularity run /home/alben/install/containers/tf2_addons.2.4.1.jupyter.fl.sif experiments_manager.py --usersettings examples/federated/mysettings_curve_fitting.py --procID 3

