'''
@author: Alexandre Benoit, LISTIC lab, FRANCE, 2022
@brief : a brief example that shows how to prepare data as SEPARATED NAMED features to feed a MODEL WITH MULTIPLE INPUTS

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
#-> apply sample preprocessing
normalizer_x1=100
normalizer_x2=10000

#-> a standard pipeline generally used for single task models
# --> we apply feature normalisation at this step
# --> here, we also name each input feature and do not merge them into a single vector/matrix
# as usually done in more complex problems
def my_preprocessing_basic_singletask_namedFeatures(single_sample_data, single_sample_label):
  #-> normalize input and create a specific
  data={'x1':single_sample_data[0]/normalizer_x1,
        'x2':single_sample_data[1]/normalizer_x2}

  #-> create
  label=single_sample_label
  return (data, label)
dataset=dataset.map(my_preprocessing_basic_singletask_namedFeatures)

#finalize pipeline with batching, shuffling and prefetching
dataset=dataset.shuffle(100).batch(batch_size).prefetch(1)


print('Tensorflow preprocessed samples dataset')
for sample in dataset:
    print(sample)

#design models for singletask:
# --> models has MULTIPLE INPUTS WITH SAME NAME as input feature names
# --> allows for automatic association and facilitates model design
def build_model_basic_singletask_namedFeatures():
    # design with multiple inputs
    x1=tf.keras.layers.Input(shape=[1], name='x1')
    x2=tf.keras.layers.Input(shape=[1], name='x2')
    X=tf.keras.layers.Concatenate(axis=1)([x1, x2])
    h=tf.keras.layers.Dense(units=10, activation='relu')(X)
    y=tf.keras.layers.Dense(units=1, activation='linear')(h)
    return tf.keras.Model(inputs=[x1, x2], outputs=y)

#allocate the target model
model=build_model_basic_singletask_namedFeatures()
model.summary()

#define loss(es)
def loss_single_task():
    return tf.keras.losses.BinaryCrossentropy()

model.compile(loss=loss_single_task())
model.fit(x=dataset,
          epochs=10,
          )
