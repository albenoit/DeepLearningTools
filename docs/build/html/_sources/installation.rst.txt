.. highlight:: shell
.. rst-class:: justify

Installation
=========================


Stable release
---------------------

No stable release yet since it's still in development.



Source files
-------------------

The sources for Deeplearningtools can be downloaded from the `Github repo`_.

.. _Github repo: https://github.com/albenoit

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/albenoit/DeepLearningTools


Virtual environment
---------------------------

Recommended installation process is to rely on containers as shown below. Frozen Python package dependency list is reported in file `requirements.txt` and is used to build containers. Then you can also perform a classical but much less reproducible and stable standard Python (Anaconda) installation using that file too. 

Container
-----------------

Relying on Singularity or Apptainer containers allows you to build the machine as a single `.sif` file and reuse (copy/paste) it on any other machine (laptop, desktop, server) where Singularity or Apptainer is installed. This is a good way to save time by avoiding multiple installation procedures, managing library conflicts, and dealing with other time-wasting tasks!

Container based installation using `Apptainer <https://apptainer.org/getting-started>`_ or `Singularity <https://sylabs.io/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Have a try with containers to get an off-the-shelf system ready to run on NVIDIA GPUs! Apptainer and Singularity will build containers from (official) Tensorflow docker images. Choose between your preferred image from the `Tensorflow docker hub <https://hub.docker.com/r/tensorflow/tensorflow/tags/>`_ or from `NVIDIA NGC <https://www.nvidia.com/en-us/gpu-cloud/containers/>`_.

I consider here Singularity or the open source fork Apptainer very close to Docker but generally more adopted for HPC. However, an equivalent container design can be done using Docker!


**install (as root):**

- `Singularity <https://sylabs.io/docs/>`
- `Apptainer (open sourced fork) <https://apptainer.org/docs>`


**build the image with GPU (as root):**

- build a custom image with the provided `install/tf2_addons.def` file that includes all python packages to build the container.
- the `install/tf_server.def` file is also provided to build a Tensorflow model server container.

::

   sudo apptainer build tf2_addons.sif tf2_addons.def #container for model training and validation
   sudo apptainer build tf_server.sif tf_server.def   #container for model serving only


**run the image (as standard user):**

- open a shell on this container, bind to your system folders of interest: ``apptainer shell --nv --bind /path/to/your/DeepLearningTools/:DeepLearningTools/ tf2_addons.sif``
- run the framework, for example on the curve fitting example: ``cd /DeepLearningTools/`` followed by ``python -m deeplearningtools.experiments_manager --usersettings examples/regression/mysettings_curve_fitting.py``
- if the GPU is not found (error such as ``libcuda reported version is: Invalid argument: expected %d.%d, %d.%d.%d, or %d.%d.%d.%d form for driver version; got "1"``, sometimes, the NVIDIA module should be reloaded after a suspend period. Recover it using command ``nvidia-modprobe -u -c=0``



More information on Apptainer (open-source fork of Singularity): `Apptainer <https://apptainer.org/getting-started>`_
