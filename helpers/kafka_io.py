# a set of helpers to experiment with kafka pubsub
# maes use of tensor serialization/parsing helpers proposed in tensor_msg_io.py

'''
# tested with kafka install and config:
wget  https://downloads.apache.org/kafka/2.7.1/kafka_2.13-2.7.1.tgz
tar -xzf kafka_2.13-2.7.1.tgz

#-> start zookeeper and kafka
./kafka_2.13-2.7.1/bin/zookeeper-server-start.sh -daemon ./kafka_2.13-2.7.1/config/zookeeper.properties
./kafka_2.13-2.7.1/bin/kafka-server-start.sh -daemon ./kafka_2.13-2.7.1/config/server.properties
echo "Waiting for 10 secs until kafka and zookeeper services are up and running"
sleep 10
#-> create the 'demo-pics' topic
./kafka_2.13-2.7.1/bin/kafka-topics.sh --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic demo-pics
#-> check topic behaviors
./kafka_2.13-2.7.1/bin/kafka-topics.sh --describe --bootstrap-server 127.0.0.1:9092 --topic demo-pics

#-> not forget to delete topic to keep disk space...
./kafka_2.13-2.7.1/bin/kafka-topics.sh  --bootstrap-server 127.0.0.1:9092 --delete --topic demo-pics

#-> list all available topics on a server:
./kafka_2.13-2.7.1/bin/kafka-topics.sh  --list --bootstrap-server 127.0.0.1:9092

#-> get topic details:
./kafka_2.13-2.7.1/bin/kafka-topics.sh  --bootstrap-server=localhost:9092 --describe --topic demo-pics
'''
import datetime
import kafka
try:
  import helpers.tensor_msg_io as tensor_msg_io
except:
  import tensor_msg_io

import tensorflow as tf
import tensorflow_io as tfio

def error_callback(exc):
    raise Exception('Error while sendig data to kafka: {0}'.format(str(exc)))

# tensorflow message publish

class KafkaIO(object):
  def __init__(self, topic_name:str, element_spec, bootstrap_servers=['localhost:9092'], flush_every=20):
    '''
      setup a kafka connexion to push Tensorflow data samples to
      topic_name:str, the log queue name to write to
      element_spec: a tuple (data, label) of tf.Tensorspec (single tensor) or dictionnaries of tf.Tensorspec (multiple named tensors)
      bootstrap_servers, the list of kafka servers(default is default local)
      flush_every, an integer that specify the iteration period when data is flushed to kafka 
    '''
    self.bootstrap_servers=bootstrap_servers
    self.topic_name=topic_name
    self.flush_every=flush_every
    self.element_spec=element_spec
    print('---> settingup connector to a kafka dataset (servers, queue)', (self.bootstrap_servers, topic_name))

    
  def kafka_producer_tf(self, items:tf.Tensor, log:str=None)->None:
    '''
    Publish tf.tensor sample pairs (tensor, label) in the form of tensorflow.Examples from an iterable (list of tensor tuples, or a dataset (WARNING, each batch will then be a sample) or something similar)
    '''
    producer=kafka.KafkaProducer(bootstrap_servers=self.bootstrap_servers)

    count=0
    
    for value, key in items:
      #-> first serialize key (say expected label/ground truth) and value (say input tensor)
      #print('value', value, type(value))
      #print('key', key, type(key))
      #tf.print('value', value)
      #tf.print('key', key)
      key_msg=tensor_msg_io.serialize_tensor(key)
      value_msg=tensor_msg_io.serialize_tensor(value)
      #print('key_msg', key_msg)
      #print('value_msg', value_msg)
      if not(count%self.flush_every):
        #producer.flush()
        print('added batchs:',count)
        #print('value_msg', value_msg)
        #print('value_msg', type(value_msg))
      #input(image_msg)
      producer.send(self.topic_name, key=key_msg, value=value_msg)
      count+=1
    """
    from multiprocessing import Pool
    print("Number of cpu : ", multiprocessing.cpu_count())
    p = Pool(multiprocessing.cpu_count()//2)
    
    def send_one_example(value, key):
      key_msg=tensor_msg_io.serialize_tensor(key)
      value_msg=tensor_msg_io.serialize_tensor(value)
      producer.send(self.topic_name, key=key_msg, value=value_msg)

    p.map(send_one_example, items)
    """
    #finalize
    metrics = producer.metrics()
    print('Producer metrics:', metrics)
    producer.flush()

    msg="\n{date} {top}, {cnt} messages written.".format(date=datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"),
                                                        top=self.topic_name,
                                                        cnt=count)
    print(msg)
    if log is not None:
      with open(log, 'a') as f:
        f.write(msg)

  def kafka_dataset_consumer_tf_basic(self, batch_size:int, shuffle:bool=False)->tfio.IODataset:
    @tf.function
    def decode_kafka_item(item):
      '''
      specify here how to decode each item. Their basic structure being to be a pair item.key, item.message, each of them being simple tensors
      Here the type of each pair element should be taken into account carefully with respect to what was done at the encoding step
      '''
      #tf.print('message', item.message)
      data = tensor_msg_io.decode_tensor_proto(item.message, dtype=self.element_spec[0].dtype)
      label =  tensor_msg_io.decode_tensor_proto(item.key, dtype=self.element_spec[1].dtype)
      #reshape to the expected shape
      data = tf.reshape(data, self.element_spec[0].shape[1:])
      label = tf.reshape(label, self.element_spec[1].shape[1:])
      #tf.print('decoded tensor, label', tf.shape(data), tf.shape(label))
      return (data, label)

    ds = tfio.IODataset.from_kafka(self.topic_name,
                                   partition=0,
                                   servers=self.bootstrap_servers[0],
                                   offset=0,
                                   configuration=["conf.topic.auto.offset.reset=earliest"])
    if shuffle is True:
      shuffle_buffer_size=10*batch_size #default setup for minimal shuffling capability on the client side
      ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.map(decode_kafka_item, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=not(shuffle))
    ds = ds.batch(batch_size).prefetch(1)
    return ds

  def kafka_dataset_consumer_tf_custom(self, features, batch_size:int, shuffle:bool=False)->tfio.IODataset:
    @tf.function
    def decode_kafka_custom_items(item):
      '''
      specify here how to decode each item. Their structure being to be a pair item.key, item.message BUT each can be a set of tensors
      Here the type of each pair element should be taken into account carefully with respect to what was done at the encoding step
      '''
      tensor = tensor_msg_io.decode_multitensor_proto(features[0], item.message)
      label =  tensor_msg_io.decode_multitensor_proto(features[1], item.key)
      #tf.print('!!!! decoded', tensor, label)#tensor=label
      for id, col in enumerate([tensor, label]): #applying same stuff for bith data and label columns
        for key in self.element_spec[id].keys():
          #print('##################################################\n[-1]+list(self.element_spec[id][key].shape[1:])=',[-1]+list(self.element_spec[id][key].shape[1:]))
          #tf.print('TO BE RESHAPED'+key, tf.shape(col[key]))
          col[key]=tf.reshape(col[key], self.element_spec[id][key].shape[1:], name='sample_'+key)
          #tf.print('RESHAPED'+key+'!!!', tf.shape(col[key]))
      #tf.print('!!!! RESHAPED', tensor, label)#tensor=label
      
      return (tensor, label)

    ds = tfio.IODataset.from_kafka(self.topic_name,
                                   partition=0,
                                   servers=self.bootstrap_servers[0],
                                   offset=0,
                                   configuration=["conf.topic.auto.offset.reset=earliest"])
    ds = ds.map(decode_kafka_custom_items, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=not(shuffle))
    if shuffle is True:
      shuffle_buffer_size=100*batch_size #default setup for minimal shuffling capability on the client side
      ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(batch_size).prefetch(batch_size)
    return ds

  def kafka_dataset_incremental_consumer_tf(self, batch_size:int, shuffle:bool=False)->tfio.IODataset:
    raise NotImplementedError('Streaming datasets not already implemented')
    online_train_ds = tfio.experimental.streaming.KafkaBatchIODataset(self.topic_name,
                                          partition=0,
                                          servers=self.bootstrap_servers[0],
                                          stream_timeout=10000, # in milliseconds, to block indefinitely, set it to -1.
                                          configuration=[
                                              "session.timeout.ms=7000",
                                              "max.poll.interval.ms=8000",
                                              "auto.offset.reset=earliest"
                                          ])
    # online_train_ds generates tiny datasets, see https://colab.research.google.com/github/tensorflow/io/blob/master/docs/tutorials/kafka.ipynb#scrollTo=9cxF0bgGkQJs
    # -> then consider incremental learning on a stream of datasets
     
#test code
if __name__ == "__main__":
	import cv2
	# tensorflow kafka demo
	#-> get demo images
	kafka_topic='demo-pics'
	from sklearn.datasets import load_sample_images
	dataset = load_sample_images()
	#print('demo images', dataset.images)
	images_tf_orig=tf.constant(dataset.images, dtype=tf.uint8)
	images_labels_orig=tf.constant(dataset.filenames, dtype=tf.string)
	print('original dataset shapes : ',images_tf_orig.shape)
	images_tf=tf.expand_dims(images_tf_orig, 1)
	images_labels=tf.expand_dims(images_labels_orig, 1)
	print('EXPECTED dataset shapes : ',images_tf.shape)
	print('images_tf', images_tf)
	demo_dataset = zip(images_tf, images_labels) # -> keep the 'key,', 'value' order to comply with the API
	#-> publish dataset to kafka
	datasample_spec=(tf.TensorSpec(shape=(1, 427, 640, 3), dtype=tf.uint8, name=None), tf.TensorSpec(shape=(), dtype=tf.string, name=None) )
	kafka_writer=KafkaIO(topic_name='demo_pics', element_spec=datasample_spec)
	kafka_writer.kafka_producer_tf(demo_dataset)
	#->read and display dataset samples
	kafka_reader=KafkaIO(topic_name='demo_pics', element_spec=datasample_spec)
	dataset_kclient=kafka_reader.kafka_dataset_consumer_tf_basic(batch_size=1)
	
	for id, sample in enumerate(dataset_kclient):
	  print('sample ', id, ': value, key shapes=', sample[0].shape, sample[1].shape)
	  cv2.imshow(str(sample[1][0].numpy()), cv2.cvtColor(sample[0][0].numpy(), cv2.COLOR_RGB2BGR))

	cv2.waitKey()

	
  # example of a custom pipeline test:
	datasample_spec=(tf.TensorSpec(shape=(1, 512, 512, 3), dtype=tf.uint8, name=None), tf.TensorSpec((1, 357, 357,1), dtype=tf.uint8, name=None) )
	kafka_reader_check=KafkaIO(topic_name='Cityscapes_hardnetmsegtrials0train', bootstrap_servers=['127.0.0.1:9092'] , element_spec=datasample_spec)#192.168.2.148:9092
	dataset_kclient_check=kafka_reader_check.kafka_dataset_consumer_tf_basic(batch_size=1)
	
	for id, sample in enumerate(dataset_kclient_check):
	  print('sample ', id, ': value, key shapes=', sample[0].shape, sample[1].shape)
	  cv2.imshow('sample'+str(id), cv2.cvtColor(sample[0][0].numpy(), cv2.COLOR_RGB2BGR))
	  cv2.imshow('label'+str(id), sample[1][0,:,:,0].numpy()*7)
	  cv2.waitKey()
	

