#/bin/sh

cd /path/to/kafka/install
kafkaFolder='kafka_2.12-3.0.0'

echo "kafka available topics TO BE DELETED:"
./$kafkaFolder/bin/kafka-topics.sh  --list --bootstrap-server 127.0.0.1:9092 > topics.list
echo "10sec before deletion..."

while read p; do
  echo "removing topic $p"
  ./$kafkaFolder/bin/kafka-topics.sh  --bootstrap-server 127.0.0.1:9092 --delete --topic $p
done <topics.list

echo "kafka topics deleted"
