.. highlight:: shell
.. rst-class:: justify
    
Features
=================

Our framework offers a full range of features to suit different needs in machine learning and data analysis. Here's an overview of the main features supported:

Ease of experiment running
------------------------------

DeepLearningTools is meant to be used as a command-line tool.




Dataset handling
--------------------






Analysis of experiments results
-----------------------------------





Reproducibility
------------------




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
