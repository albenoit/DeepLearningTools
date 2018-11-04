# python 2&3 compatibility management
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import tensorflow as tf
import numpy as np

from tensorflow.contrib.keras import layers
from tensorflow.python.layers.core import Dense


#CONCRETE DROPOUT from Yarin Gal & al : https://arxiv.org/abs/1705.07832
class ConcreteDropout(layers.Wrapper):
    """This wrapper allows to learn the dropout probability
        for any given input layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$
             (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and
             N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eucledian
            loss.

    # Warning
        You must import the actual layer class from tf layers,
         else this will not work.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = (np.log(init_min) - np.log(1. - init_min))
        self.init_max = (np.log(init_max) - np.log(1. - init_max))

    def build(self, input_shape=None):
        self.input_spec = layers.InputSpec(shape=input_shape)
        if hasattr(self.layer, 'built') and not self.layer.built:
            self.layer.build(input_shape)

        # initialise p
        self.p_logit = self.add_variable(name='p_logit',
                                         shape=None,
                                         initializer=tf.random_uniform(
                                             (1,),
                                             self.init_min,
                                             self.init_max),
                                         dtype=tf.float32,
                                         trainable=True)
        self.p = tf.nn.sigmoid(self.p_logit[0], name='dropout_rate')
        tf.summary.scalar('dropoutRate', self.p)
        tf.add_to_collection("LAYER_P", self.p)

        # initialise regulariser / prior KL term
        input_dim = int(np.prod(input_shape[1:]))

        weight = self.layer.kernel
        with tf.name_scope('dropout_regularizer'):
            kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(
                weight)) / (1. - self.p)
            dropout_regularizer = self.p * tf.log(self.p)
            dropout_regularizer += (1. - self.p) * tf.log(1. - self.p)
            dropout_regularizer *= self.dropout_regularizer * input_dim
            regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
            # Add the regularisation loss to collection.
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        with tf.name_scope('dropout_on_input'):
            eps = 1e-7
            temp = 0.1

            unif_noise = tf.random_uniform(shape=tf.shape(x))
            drop_prob = (
                tf.log(self.p + eps)
                - tf.log(1. - self.p + eps)
                + tf.log(unif_noise + eps)
                - tf.log(1. - unif_noise + eps)
            )
            drop_prob = tf.nn.sigmoid(drop_prob / temp)
            random_tensor = 1. - drop_prob

            retain_prob = 1. - self.p
            x *= random_tensor
            x /= retain_prob
        return x

    def call(self, inputs, training=None):
        #if training:
        return self.layer.call(self.concrete_dropout(inputs))
        #else:
        #    return self.layer.call(inputs)


def concrete_dropout(inputs, layer,
                     trainable=True,
                     weight_regularizer=1e-6,
                     dropout_regularizer=1e-5,
                     init_min=0.1, init_max=0.1,
                     training=True,
                     name=None,
                     **kwargs):

    cd_layer = ConcreteDropout(layer, weight_regularizer=weight_regularizer,
                               dropout_regularizer=dropout_regularizer,
                               init_min=init_min, init_max=init_max,
                               trainable=trainable,
                               name=name)
    return cd_layer.apply(inputs, training=training)



def track_gradients(loss):
  ''' Helper function to use in the getOptimizer function
      to compute and gradients and log them into the Tensorboard
      Args:
        loss: the loss to be appled for gradients computation
      Returns:
        tvars: the trainable variables
        raw_grads: the raw gradient values
        gradient_norm: the gradient global norm
  '''
  #get all trained variables that will be optimized
  tvars = tf.trainable_variables()
  #compute gradients and track them
  raw_grads = tf.gradients(loss, tvars)
  #track gradient global norm
  gradient_norm=tf.global_norm(raw_grads, name='loss_grad_global_norm')
  tf.summary.scalar('gradient_global_norm_raw', gradient_norm)
  for grad in raw_grads:
    if grad is not None:
      trainable_nb_values=np.prod(grad.get_shape().as_list())
      if trainable_nb_values>1:
          tf.summary.histogram(grad.op.name, grad)
  return tvars, raw_grads, gradient_norm
