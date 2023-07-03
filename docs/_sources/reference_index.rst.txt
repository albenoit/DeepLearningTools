.. highlight:: shell
.. rst-class:: justify
    
DeepLearningTools package
===========================

Main scripts
-----------------

.. toctree::
   :maxdepth: 4

   main_script


Test scripts
--------------------

The test scripts include unit tests for the framework, covering various scenarios such as basic training, regression, classification, 
Flower simulation, Kafka use, tensor operations, and time series.

.. toctree::
   :maxdepth: 4

   unit_test

Helpers subpackage
--------------------------

The "deeplearningtools.helpers" sub-package includes several modules containing functions and utility classes for different tasks in the field of deep learning. 
These modules offer features such as network alignment, attention, network distances, federated processing, IO file management, Kafka IO management, 
loss functions, modeling and deployment tools, OpenCV tools, tensor management and TFRecords files.

.. toctree::
   :maxdepth: 4

   reference_helpers


Tools subpackage
-------------------------

The "deeplearningtools.tools" sub-package contains several modules that offer useful features and tools for deep learning tasks. 
These modules include custom callbacks for models, a command-line parser, classes for experiment parameters, 
functions for loading experiment parameters and model definitions, and a tool for inserting additional hyperparameters into experiment parameters.

.. toctree::
   :maxdepth: 4

   reference_tools
