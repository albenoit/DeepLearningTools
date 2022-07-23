'''
@author: Alexandre Benoit, LISTIC lab, FRANCE, 2022
@brief : a brief example that shows how to prepare data and model to conduct
multitask learning

WARNING, only the training dataset pipeline is prepared here for demo purpose
=> complete model optimization will introduce a similar pipeline but relying on a validation dataset!

Look for more advanced pipelines (image crops, csv file preprocessing) in script DataProvider_input_pipeline.py in the root directory
'''
import numpy as np
import tensorflow as tf
#data pipe draft
batch_size=3
 #-> consider a tuple of x, y samples to be considered as a dataset
x1=np.random.randint(100, size=(10))
x2=x1**2
X=np.array([x1,x2]).transpose() #-> let's merge the 2 inputs into a single table as usually done
#print(X.shape)
y=x2>100
samples=(X, y)
#basic visualisation
print('samples, raw python', samples)
#->create the related tensorflow dataset
dataset=tf.data.Dataset.from_tensor_slices(samples)
"""#basic visualisation
print('Tensorflow raw samples dataset')
for sample in dataset:
    print(sample)
"""
#-> apply a more advanced sample preprocessing adapted to multitask learning:
#    given 2 tasks, for instance 1) autoencoding, 2) classification
# --> we show how to set labels as named values (dictionaries)
# to comply with multitask learning optimization
normalizer_x1=100
normalizer_x2=10000
def my_preprocessing_advanced_multitask(single_sample_data, single_sample_label):
  #-> normalize input and create a specific
  data=tf.stack([single_sample_data[0]/normalizer_x1,
                single_sample_data[1]/normalizer_x2],
                axis=0)

  #-> create
  label={'class':single_sample_label, 'ae':data }
  return (data, label)
dataset=dataset.map(my_preprocessing_advanced_multitask)

#finalize pipeline with batching, shuffling and prefetching
dataset=dataset.shuffle(100).batch(batch_size).prefetch(1)


print('Tensorflow preprocessed samples dataset')
for sample in dataset:
    print(sample)

#design models for multitask:
# multiples model outputs are defined 
# --> WITH OUTPUT NAMES MATCHING INPUT LABEL FEATURE NAMES here 'class' and 'ae'!!! 

def build_model_advanced_multitask():
    # design with multiple outputs
    x=tf.keras.layers.Input(shape=[2])
    h=tf.keras.layers.Dense(units=10, activation='relu')(x)
    y=tf.keras.layers.Dense(units=1, activation='linear', name='class')(h) #classification head
    ae=tf.keras.layers.Dense(units=2, activation='linear', name='ae')(h) #autoencoding head
    return tf.keras.Model(inputs=[x], outputs=[y, ae])

#allocate the target model
model=build_model_advanced_multitask()
model.summary()


#define losses
def loss_multi_task():
  # one defines one loff for each output
  # NAMES MUST MATCH : model output name AND label name
  return {'class':tf.keras.losses.BinaryCrossentropy(),
          'ae':tf.keras.losses.MeanAbsoluteError(),
          }


model.compile(loss=loss_multi_task())
model.fit(x=dataset,
          epochs=10,
          )
