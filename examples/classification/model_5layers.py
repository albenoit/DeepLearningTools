""" basic convolutionnal model inspired from https://www.tensorflow.org/tutorials/images/classification
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

def model(usersettings):
  model = Sequential([
      Conv2D(16, 3, padding='same', activation='relu',
             input_shape=(usersettings.hparams['imgHeight'], usersettings.hparams['imgWidth'] ,3)),
      MaxPooling2D(),
      Dropout(usersettings.hparams['dropout']),
      Conv2D(32, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Conv2D(64, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Dropout(usersettings.hparams['dropout']),
      Flatten(),
      Dense(512, activation='relu'),
      Dense(1, activation='sigmoid')
  ])

  return model
