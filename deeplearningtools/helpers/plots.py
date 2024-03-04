# ========================================
# FileName: model.py
# Date: 13 july 2023 - 22:49
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of custom ploting tools
# for DeepLearningTools.
# =========================================
import io
import numpy as np
import itertools
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg') # desactivating X server mode, required in simulation mode
import matplotlib.pyplot as plt

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call.
  from https://www.tensorflow.org/tensorboard/image_summaries?hl=fr
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def plot_confusion_matrix(cm, normalized=False, class_names=None):
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
  labels = cm if normalized is False else np.around(np.divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis]+1e-6), decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure 