# Intaracting with kafka to load data
Instead of loading data from the datapipeline defined in your experiment settings (and burn calories for each trial reloading the same data), user may prefer relying on publish subscribe like systems.
A Kafka connector is natively proposed in Tensorflow, the proposed framework can make use of it transparently.

The main idea is to propose a script *start_kafka_producer.py* that makes use of the user defined datapipeline to upload data to a kafka server. Next, by the activation of the kafka option in the experiment settings file, model can be trained from this kafka feed.

In the following are drawn the main steps to build kafka containers using docker-compose or singularity-compose.

# Install/build containers
chdir to the folder where file docker-compose.yml and singularity-compose.yml are defined, here:
  `cd /path/to/listic-deeptool/install/kafka`

Then build containers using commandline using either singularity-compose or docker-compose :

  `singularity-compose build`

  or

  `docker-compose build`

# run containers 
Launch from commandline:

  `docker-compose up` or `singularity-compose up`

Check running instances:

  `singularity-compose ps`

Result example:

  `
INSTANCES  NAME         PID     IP              IMAGE
1          kafka        271602  10.22.0.3       kafka.sif
2      zookeeper        271333  10.22.0.2       zookeeper.sif
  `

## USE
### Uploading data to kafka
Upload data to kafka using the datapipeline defined in an experiment settings file using the *start_kafka_producer.py* python script. For instance:

`singularity run --nv install/tf2_addons.2.8.0.sif start_kafka_producer.py --server localhost:9092 --procID 0 --usersettings examples/regression/mysettings_curve_fitting.py`

### Reading data from kafka
By default, user rely on the datapipeline you defined in the experiment settings file. Data is then loaded online as defined in this fucntion.

Now considering that the same experiment settings file has been used to upload data to kafka as described just above, experiments can be conducted by reading from kafka. This opens to online learning and/or load data more rapidly (preprocessing can be done before, at the upload step).

Then, for a given experiment, in order to switch to kafka pipelining, user just has to set  the following variables in the experiment settings file:

  `consume_data_from_kafka=True`

  and specify server(s) IP (adjust the following line as required)
  
  `kafka_bootstrap_servers=['localhost:9092']`

Enjoy !