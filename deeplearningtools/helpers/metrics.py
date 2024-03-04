# ========================================
# FileName: model.py
# Date: 13 july 2023 - 22:43
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of custom metrics used along the training and validation stages along model fitting
# for DeepLearningTools.
# maybe have a look there to compare/check for errors: https://neptune.ai/blog/keras-metrics
# =========================================

import io
import numpy as np
from deeplearningtools.helpers import plots
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# From a proposal and refactoring performed by Google Bard that referenced https://github.com/Zs0819/DR_detection_and_HAR
# ... and with many fixes
class ConfusionMatrix(tf.keras.metrics.Metric):
    """
    Custom Metric able to compute the Confusion matrix in a classification problem
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
        """#DEBUG PURPOSE : tf.print('conf mat', tf.math.confusion_matrix(y_true, y_pred))
        tf.print('y_true', y_true)
        tf.print('y_pred', y_pred)
        tf.print(tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.dtypes.int64))
        tf.print('self.confusion_matrix',self.confusion_matrix)
        """
        self.confusion_matrix.assign_add(tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.dtypes.int64))

    def result(self):
        acc=tf.math.reduce_sum(tf.linalg.tensor_diag_part(self.confusion_matrix))/tf.math.reduce_sum(self.confusion_matrix)
        
        FP = self.confusion_matrix[0][1]
        FN = self.confusion_matrix[1][0]
        TP = self.confusion_matrix[0][0]

        recall = TP/(TP+FN)
        precision = TP/(TP+FP)

        precision = precision
        recall = recall

        return {'Accuracy':acc, "Precision": precision, "Recall": recall}

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros([self.num_classes, self.num_classes], dtype=tf.dtypes.int64))
        
    def get_confusion_matrix_numpy(self):
        return self.confusion_matrix.numpy()

    def get_image_summary(self):
        figure=plots.plot_confusion_matrix(self.get_confusion_matrix_numpy())
        cm_image = plots.plot_to_image(figure)
        return cm_image

    """
    def non_nan_average(self, x):
        # Computes the average of all elements that are not NaN in a rank 1 tensor
        nan_mask = tf.math.is_nan(x)
        x = tf.boolean_mask(x, tf.logical_not(nan_mask))
        return tf.mean(x)
    """
    def get_accuracy(self):
        # Computes the average of all elements that are not NaN in a rank 1 tensor
        diag = tf.linalg.tensor_diag_part(self.confusion_matrix)    

        # Calculate the total number of data examples for each class
        total_per_class = tf.reduce_sum(self.confusion_matrix, axis=1)
        acc_per_class = diag / tf.maximum(1, total_per_class)
        # @bug ? if accuracy is 1.0 for a class, it might return a NaN value (division by 0)
        return acc_per_class

    def gen_stats(self):
        """Create a pyplot plot and save to buffer."""



        print(self.confusion_matrix)
        FP = self.confusion_matrix[0][1]
        FN = self.confusion_matrix[1][0]
        TP = self.confusion_matrix[0][0]

        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        #accuracy = self.get_accuracy()     

        fig, ax = plt.subplots()

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        #accuracy = accuracy.numpy()
        precision = precision.numpy()
        recall = recall.numpy()
        
        data = np.concatenate(([precision], [recall]), axis=0)
        print(data)
        #print({'Acc':round(acc,2), 'Precision':round(precision.numpy(),2), 'Recall':round(recall.numpy(),2)})
        df = pd.DataFrame(data, index=["Precision", "Recall"])

        print(df)
        print("FP : ", FP)
        print("FN : ", FN)
        print("TP : ", TP)

        ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
        
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        return image
        



class BinaryConfusionMatrix(ConfusionMatrix):
    """
    Custom Metric able to compute the Confusion matrix in a binary classification problem
    This call computes the confusion matrix that can be retreived as an image from the get_specific_summary method
    that can be intergrated in Tensorboard using a callback (see deeplearningtools.callbacks)
    As a standard metric that reports a scalar value, this class summarises the confusion matrix with the accuracy metric
    """
    def __init__(self, name='confusion_matrix'):
        super(BinaryConfusionMatrix, self).__init__(num_classes=2, name=name)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int64), [-1])
        y_pred = tf.reshape(tf.cast(tf.greater(y_pred, 0.5), tf.int64), [-1])
        self.confusion_matrix.assign_add(tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.dtypes.int64))

class BiasMetrics(tf.keras.metrics.Metric):
    """
    Custom Metric able to compute the Bias metrics in a classification problem
    This call computes the typical bias metrics:
    - Statistical Parity Difference (SPD)
    - Disparate Impact (DI)
    - Equal Opportunity Difference (EOD)
    - Average Absolute Odds Difference (AAOD)
    - Error Rate Difference (ERD)
    """
    def __init__(self, sensitive_column_index, name='bias_metrics'):
        super(BiasMetrics, self).__init__(name)
        self.sensitive_column_index = sensitive_column_index
        self.count_and_p1y1s1=tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_and_p1y1s0=tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_and_p1y0s0=tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_and_p1y0s1=tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_and_p0y1s0=tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_and_p0y1s1=tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_and_y0s0=tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_and_y0s1=tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_and_y1s0=tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_and_y1s1=tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_unprivileged_p1s0=tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_privileged_s1 =tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_unprivileged_s0 =tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.count_privileged_p1s1 =tf.Variable(tf.constant(0, tf.int64), trainable=False)

        self.tmp_sensitive_data =tf.Variable(tf.constant(0, tf.int64), trainable=False)
        self.tmp_non_sensitive_data =tf.Variable(tf.constant(0, tf.int64), trainable=False)


    def update_state(self, y_true, y_pred, sample_weight=None):
        # 1. Prepare low level indicators
        # binarize y_pred 
        y_pred_labels = tf.math.greater(y_pred[:, 0], 0.5)
        not_y_pred_labels = tf.math.logical_not(y_pred_labels)
        # get labels
        y_true_labels = tf.math.greater(y_true[:, 0], 0)
        tf.print(y_pred_labels)
        not_y_true_labels = tf.math.logical_not(y_true_labels)
        # get target sensitive attribute
        sensitive_attribute = tf.math.greater(y_true[:, self.sensitive_column_index], 0)
        not_sensitive_attribute = tf.math.logical_not(sensitive_attribute)
        
        # @debug - supprimer ces 2 lignes
        sensitive_attribute_TMP = tf.math.reduce_sum(tf.cast(sensitive_attribute, tf.int64), 0)
        self.tmp_sensitive_data.assign_add(sensitive_attribute_TMP)
        non_sensitive_attribute_TMP = tf.math.reduce_sum(tf.cast(not_sensitive_attribute, tf.int64), 0)
        self.tmp_non_sensitive_data.assign_add(non_sensitive_attribute_TMP)


        # Calculate the number of elements where Pr(y_batch_np = 1, y_pred_batch_np = 1,sensitive_attributes_np = 1)
        pred_and_labels = y_pred_labels & y_true_labels # tf.math.logical_and(y_pred_labels, y_true_labels)
        # 2. cumulate conditioned counts
        self.count_and_p1y1s1.assign_add(tf.math.reduce_sum(tf.cast(pred_and_labels & sensitive_attribute, tf.int64)))
        # Calculate the number of elements where P(y_batch_np = 1, y_pred_batch_np = 1,  sensitive_attributes_np = 0)
        self.count_and_p1y1s0.assign_add(tf.math.reduce_sum(tf.cast(pred_and_labels & not_sensitive_attribute, tf.int64)))
        #Pr ( Ŷ = 1,Y = 0, S = 0)
        self.count_and_p1y0s0.assign_add(tf.math.reduce_sum(tf.cast(y_pred_labels & not_y_true_labels & not_sensitive_attribute, tf.int64)))
        #Pr ( Ŷ = 1,Y = 0, S = 1)
        self.count_and_p1y0s1.assign_add(tf.math.reduce_sum(tf.cast(y_pred_labels & not_y_true_labels & sensitive_attribute, tf.int64)))
        #Pr ( Ŷ = 0,Y = 1, S = 0)
        self.count_and_p0y1s0.assign_add(tf.math.reduce_sum(tf.cast(not_y_pred_labels & y_true_labels & not_sensitive_attribute, tf.int64)))
        #Pr ( Ŷ = 0,Y = 1, S = 1)
        self.count_and_p0y1s1.assign_add(tf.math.reduce_sum(tf.cast(not_y_pred_labels & y_true_labels & sensitive_attribute, tf.int64)))
        #pr( y=0,s=0)
        self.count_and_y0s0.assign_add(tf.math.reduce_sum(tf.cast(not_y_true_labels & not_sensitive_attribute, tf.int64)))
        #pr( y=0,s=1)
        self.count_and_y0s1.assign_add(tf.math.reduce_sum(tf.cast(not_y_true_labels & sensitive_attribute, tf.int64)))
        #pr( y=1,s=0)
        self.count_and_y1s0.assign_add(tf.math.reduce_sum(tf.cast(y_true_labels & not_sensitive_attribute, tf.int64)))
        #pr( y=1,s=1)
        self.count_and_y1s1.assign_add(tf.math.reduce_sum(tf.cast(y_true_labels & sensitive_attribute, tf.int64)))
        #pr (ŷ=1,s=1)
        self.count_privileged_p1s1.assign_add(tf.math.reduce_sum(tf.cast(y_pred_labels & sensitive_attribute, tf.int64)))
        #pr (ŷ=1,s=0)
        self.count_unprivileged_p1s0.assign_add(tf.math.reduce_sum(tf.cast(y_pred_labels & not_sensitive_attribute, tf.int64)))
        self.count_privileged_s1.assign_add(tf.math.reduce_sum(tf.cast(sensitive_attribute, tf.int64)))
        self.count_unprivileged_s0.assign_add(tf.math.reduce_sum(tf.cast(not_sensitive_attribute, tf.int64)))
        
    def result(self):
        # compute bias metrix for the epoch
        # tf.maximum(value, 1) to avoid division by 0
        unprivileged_win_prob= tf.cast(self.count_unprivileged_p1s0, tf.float64)/tf.cast(self.count_unprivileged_s0, tf.float64)
        privileged_win_prob= tf.cast(self.count_privileged_p1s1, tf.float64)/tf.cast(self.count_privileged_s1, tf.float64)
        fpr_unpriviledged= tf.cast(self.count_and_p1y0s0, tf.float64)/tf.cast(self.count_and_y0s0, tf.float64)
        fpr_priviledged  = tf.cast(self.count_and_p1y0s1, tf.float64)/tf.cast(self.count_and_y0s1, tf.float64)
        fnr_unpriviledged= tf.cast(self.count_and_p0y1s0, tf.float64)/tf.cast(self.count_and_y1s0, tf.float64)
        fnr_priviledged  = tf.cast(self.count_and_p0y1s1, tf.float64)/tf.cast(self.count_and_y1s1, tf.float64)

        fpr_diff= fpr_unpriviledged - fpr_priviledged
        tpr_diff=tf.math.divide_no_nan(tf.cast(self.count_and_p1y1s0, tf.float64), tf.cast(self.count_and_y1s0, tf.float64)) - tf.math.divide_no_nan(tf.cast(self.count_and_p1y1s1, tf.float64), tf.cast(self.count_and_y1s1, tf.float64))             
        
        return {self.name+'_SPD': unprivileged_win_prob - privileged_win_prob,
                self.name+'_DI': unprivileged_win_prob / privileged_win_prob,
                self.name+'_EOD': tf.cast(self.count_and_p1y1s0, tf.float64)/tf.cast(self.count_and_y1s0, tf.float64) - tf.cast(self.count_and_p1y1s1, tf.float64)/tf.cast(self.count_and_y1s1, tf.float64),
                self.name+'_AAOD': 0.5*(fpr_diff+tpr_diff),
                self.name+'_ERD':(fpr_unpriviledged+fnr_unpriviledged)-(fpr_priviledged+fnr_priviledged),
                } 
    def test_results(self):
        results_base=self.result()
        #get intermediate values for debugging
        addons={'count_and_y0s0':self.count_and_y0s0,
                'count_and_y0s1':self.count_and_y0s1,
                'count_and_y1s0':self.count_and_y1s0,
                'count_and_y1s1':self.count_and_y1s1,
                #'self.count_unprivileged_p1s0':self.count_unprivileged_p1s0,
                #'self.count_unprivileged_s0':self.count_unprivileged_s0,
                #'self.count_privileged_p1s1':self.count_privileged_p1s1,
                #'self.count_privileged_s1':self.count_privileged_s1,
        }
        print('results_base', results_base)
        print('addons', addons)
        results_base.update(addons)
        print('all_res', results_base)
        return results_base

    def reset_state(self):
        self.count_and_p1y1s1.assign(tf.constant(0, tf.int64))
        self.count_and_p1y1s0.assign(tf.constant(0, tf.int64))
        self.count_and_p1y0s0.assign(tf.constant(0, tf.int64))
        self.count_and_p1y0s1.assign(tf.constant(0, tf.int64))
        self.count_and_p0y1s0.assign(tf.constant(0, tf.int64))
        self.count_and_p0y1s1.assign(tf.constant(0, tf.int64))
        self.count_and_y0s0.assign(tf.constant(0, tf.int64))
        self.count_and_y0s1.assign(tf.constant(0, tf.int64))
        self.count_and_y1s0.assign(tf.constant(0, tf.int64))
        self.count_and_y1s1.assign(tf.constant(0, tf.int64))
        self.count_unprivileged_p1s0.assign(tf.constant(0, tf.int64))
        self.count_privileged_s1.assign(tf.constant(0, tf.int64))
        self.count_unprivileged_s0.assign(tf.constant(0, tf.int64))
        self.count_privileged_p1s1.assign(tf.constant(0, tf.int64))
