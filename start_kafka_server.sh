#/bin/sh


##### Start a zookeeper and Kafka server on the local machine
# tested with kafka install and config installed, say here : /home/alben/install/pubsub
#cd /home/alben/install/pubsub
#wget  https://downloads.apache.org/kafka/2.7.1/kafka_2.13-2.7.1.tgz
#tar -xzf kafka_2.13-2.7.1.tgz
#wget https://downloads.apache.org/kafka/3.1.0/kafka_2.13-3.1.0.tgz
cd /home/alben/install/pubsub
echo "Starting Zookeeper and Kafka server, WARNING, TO BE LAUNCHED FROM THE PARENT FOLDER OF A KAFKA INSTALL"
export kafkaFolder='kafka_2.13-3.1.0'
#kafka_2.13-2.7.1'

#-> start zookeeper and kafka ... TO BE LAUNCHED FROM THE PARENT FOLDER OF A KAFKA INSTALL
./$kafkaFolder/bin/zookeeper-server-start.sh -daemon ./$kafkaFolder/config/zookeeper.properties
./$kafkaFolder/bin/kafka-server-start.sh -daemon ./$kafkaFolder/config/server.properties
echo "Waiting for 10 secs until kafka and zookeeper services are up and running"
sleep 10
#-> list all available topics on a server:
echo "Kafka server started, already available logs:"
./$kafkaFolder/bin/kafka-topics.sh  --list --bootstrap-server 127.0.0.1:9092
echo "ready to go!"

# LOGS LOCATION : by default, queues/logs are stored here : /tmp/kafka-logs

###### other commands of interest :
# -> check topic behaviors (ex: demo-pics)
#./$kafkaFolder/bin/kafka-topics.sh --describe --bootstrap-server 127.0.0.1:9092 --topic demo-pics

#-> not forget to delete topic to keep disk space...
#./$kafkaFolder/bin/kafka-topics.sh  --bootstrap-server 127.0.0.1:9092 --delete --topic demo-pics


#-> get topic details:
#./$kafkaFolder/bin/kafka-topics.sh  --bootstrap-server=localhost:9092 --describe --topic demo-pics
