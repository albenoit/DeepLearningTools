.. highlight:: shell
.. rst-class:: justify
    
Features
=================

Our framework offers a full range of features to suit different needs in machine learning and data analysis. Here's an overview of the main features supported.

Ease of experiment running
------------------------------

DeepLearningTools is meant to be used as a command-line tool.

    - A single script/module, `experiments_manager.py`, that manages all the train/val/export/serve process is provided to let you no more care about it.

    - You write the experiment settings file that focuses on the experiment but avoid the machinery. You then define the expected variables and functions (datasets, learning rates, loss, etc.). This is enough work but only focused on the experiment.

    - You write your model in a separate file following a basic function prototype. This will allow you to switch between models but still relying on the same experiment settings.

Dataset handling
--------------------

`In progress ...`


Analysis of experiments results
-----------------------------------

For each experiment, a file (named by experiment type, hyperparameters and times) is produced, grouping together several metadata informations. It contains, among other things, model outputs, logs and a copy of the scripts used.

You can regularly look at the Tensorboard to monitor indicators, weight distributions, model output embedding, etc.

Once trained (or along training), start the Tensorboard parsing logs of the experiments folder (provided example is experiments/examples/curve_fitting) using:

.. code-block:: shell

    tensorboard  --logdir=experiments/examples/curve_fitting

Then, open a web browser and reach http://127.0.0.1:6006/ to monitor training values and observe the obtained embedding.

Reproducibility
------------------

Reproducibility of results is ensured by the use of containers which freeze the software environment. Nevertheless, calculation times will still depend on hardware resources. 


Use of additional tools
---------------------------

- The program can be used with high-level tools such as `hyperopt <https://hyperopt.github.io/>`_ to explore hyperparameter space. Examples and demonstrations are available in the directory ``examples/hyperopt``.
- Federated learning is supported by the system through the `Flower <https://flower.dev/>`_ library. With just a few modifications, you can switch from your classic training model to the federated version ``examples/federated``.


Roadmap
-----------
* [x] Designing the overall system architecture
* [x] Implementing core functionalities
* [x] Conducting unit tests to ensure proper functioning
* [x] Federated Learning compliant
* [x] Explainable compliant, check "ICPR-XAIE2022" (https://hal.archives-ouvertes.fr/hal-03719597)

* [ ] Creating a user-friendly web-based dashboard to simplify framework interaction
* [ ] Implementing a robust security system to protect sensitive user data
* [ ] Optimizing resource utilization parameters for maximum efficiency (GPU/TPU/embedded systems)
* [ ] Extending federated learning with privacy-preserving techniques or adaptive aggregation methods
* [ ] Extending XAI techniques to help understand model predictions and decision making
