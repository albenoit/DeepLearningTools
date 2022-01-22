#/bin/sh

cd /path/to/kafka/install
kafkaFolder='kafka_2.12-3.0.0'

echo "kafka available topics:"
./$kafkaFolder/bin/kafka-topics.sh  --list --bootstrap-server 127.0.0.1:9092 > topics.list
