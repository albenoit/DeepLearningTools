# ========================================
# FileName: model.py
# Date: 13 july 2023 - 22:43
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of custom metrics used along the training and validation stages along model fitting
# for DeepLearningTools.
# =========================================

import io
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from deeplearningtools.helpers import plots
from tensorflow.keras.metrics import Metric

def plot_confusion_matrix(cm, class_names=None):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.
  from https://www.tensorflow.org/tensorboard/image_summaries?hl=fr
  
  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
      
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  if isinstance(class_names, list):
    if len(class_names)==cm.shape[0]:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(np.divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis]), decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

# From a proposal and refactoring performed by Google Bard that referenced https://github.com/Zs0819/DR_detection_and_HAR
# ... and with many fixes
class ConfusionMatrix(tf.keras.metrics.Metric):
    """
    Custom Metric able to compute the COnfusion matrix in a classification problem
    This call computes the confusion matrix that can be retreived as an image from the get_specific_summary method
    that can be intergrated in Tensorboard using a callback (see deeplearningtools.callbacks)
    As a standard metric that reports a scalar value, this class summarises the confusion matrix with the accuracy metric
    """
    def __init__(self, num_classes, name='confusion_matrix'):
        super(ConfusionMatrix, self).__init__(name)
        self.num_classes = int(num_classes)
        self.confusion_matrix = tf.Variable(
            tf.zeros([self.num_classes, self.num_classes], dtype=tf.int64),
            trainable=False,
            dtype=tf.int64
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int64), [-1])
        y_pred = tf.reshape(tf.math.argmax(y_pred, axis=-1, output_type=tf.int64), [-1])
        #DEBUG PURPOSE : tf.print('conf mat', tf.math.confusion_matrix(y_true, y_pred))
        self.confusion_matrix.assign_add(tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.dtypes.int64))

    def result(self):
        acc=tf.math.reduce_sum(tf.linalg.tensor_diag_part(self.confusion_matrix))/tf.math.reduce_sum(self.confusion_matrix)
        return {'Accuracy':acc} 

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros([self.num_classes, self.num_classes], dtype=tf.dtypes.int64))
        
    def get_image_summary(self):
        figure=plot_confusion_matrix(self.confusion_matrix.numpy())
        cm_image = plots.plot_to_image(figure)
        return cm_image
