.. highlight:: shell

Frequently asked questions
=============================

Unable to load a module.
********************************
Question 1: Why I have this error?

.. code-block:: console
    
    $ :/usr/bin/python3 : Error while finding module specification for 
    'deeplearningtools.experiments_manager' (ModuleNotFoundError : No module named 'deeplearningtools')

Answer 1: Bind mounts use.

When a container is launched, it has its own file system, usually based on a base image. 
By default, the working directory in the container is usually the root directory (/) of the container file system. 
However, to facilitate interaction with the host, it is common practice to bind a host directory to a container directory.
The bind mount allows a host directory to be mounted in a specified container location. 
This allows the application inside the container to access files and resources on the host, while maintaining the container's isolation. 
A bind is established by adding these parameters below to the command:

.. code-block:: console
    
    apptainer shell --nv --bind /path/to/your/DeepLearningTools/:DeepLearningTools/ tf2_addons.sif
    #or
    apptainer run --nv --bind /path/to/your/DeepLearningTools/:DeepLearningTools/ tf2_addons.sif -m deeplearningtools.experiments_manager -u examples/regression/mysettings_curve_fitting.py
