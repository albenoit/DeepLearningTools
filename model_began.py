'''
Initial code From https://github.com/khanrc/tf.gans-comparison
Adapted to the experiment manager framework
'''
# coding: utf-8
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
from tensorflow.contrib.learn import ModeKeys

class Began(object):
    def __init__(self, data,
                n_outputs,
                hparams #TODO use it to make the model customisable by external hyperparameters
                ):
                #D_lr=1e-4, G_lr=1e-4, image_shape=[64, 64, 3], generator_code_dim=64, gamma=0.5):
        self.name='BEGAN'
        self.generator_code_dim=n_outputs
        self.input=data
        self.nf = 64

    def get_model_graph(self, mode):
        ''' switch between train or served graph depending on the processing mode
        Args:
        mode: processing mode keyword from tensorflow.contrib.learn import ModeKeys
        Returns: the target graph
        '''
        is_training = mode == ModeKeys.TRAIN
        if mode!=ModeKeys.INFER:
            return self.get_train_val_graph(is_training)
        else:
            return self.get_served_graph()

    def get_served_graph(self):
        '''the subpart of the train graph is build here for model serving
        Returns: the served graph model
        '''
        with tf.variable_scope(self.name):
            G = self._generator(self.input)
            # Generator of BEGAN does not use tanh activation func.
            # So the generated sample (fake sample) can exceed the image bound [-1, 1].
            fake_sample = tf.clip_by_value(G, -1., 1.)

            return {'generator_fake_samples':fake_sample}
        print('TODO')


    def get_train_val_graph(self, is_training):
        '''the complete graph used for training is built here
        Args:
            is_training:boolean, True if training, False for validation
        Returns: the train/val graph model
        '''
        with tf.variable_scope(self.name):
            real_sample_batch=self.input
            batch_size=tf.shape(real_sample_batch)[0]
            generator_input_code= tf.random_uniform(shape=[batch_size,self.generator_code_dim],
                                                    minval=-1.,
                                                    maxval=1.,
                                                    dtype=tf.float32)
            '''
            generator_input_code= tf.truncated_normal(
                                            shape=[batch_size,self.generator_code_dim],
                                            mean=0.0,
                                            stddev=0.5)
            '''
            tf.summary.histogram('code_gen',generator_input_code)
            print('Real data input :'+str((real_sample_batch,real_sample_batch.graph)))
            print('Generator input :'+str((generator_input_code,generator_input_code.graph)))

            G = self._generator(generator_input_code)
            print('INFO: Generator output tensor :'+str((G,G.graph)))

            # Discriminator is not called an energy function in BEGAN. The naming is from EBGAN.
            D_real_energy, D_real_code = self._discriminator(real_sample_batch, name='D_real_energy')
            D_fake_energy, D_fake_code = self._discriminator(G, reuse=True, name='D_fake_energy')
            print('INFO: D_real_energy tensor='+str(D_real_energy))
            print('INFO: D_fake_energy tensor='+str(D_fake_energy))
            return {'generator_fake_samples':G,
                    'D_real_energy':D_real_energy,
                    'D_fake_energy':D_fake_energy,
                    'D_real_code':D_real_code,
                    'D_fake_code':D_fake_code,
                    }

    def _encoder(self, X, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            print('### Encoder Input ='+str(X))

            nf = self.nf
            with slim.arg_scope([slim.conv2d], kernel_size=[3,3], padding='SAME', activation_fn=tf.nn.elu):
                net = slim.conv2d(X, nf)
                print('FIRST Encoder layer ='+str(net))

                #net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf*2, stride=2) # 28x28=>14*14
                print('Encoder layer ='+str(net))

                #net = slim.conv2d(net, nf*2)
                net = slim.conv2d(net, nf*3, stride=2) # 14*14=>7*7
                print('Encoder layer ='+str(net))
                #net = slim.conv2d(net, nf*3)
                net = slim.conv2d(net, nf*2)

            net = slim.flatten(net)
            print('Encoder layer ='+str(net))
            h = slim.fully_connected(net, self.generator_code_dim, activation_fn=None)

            print('### Encoder OUTPUT ='+str(h))

            return h

    def _decoder(self, h, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            print('*** Decoder INPUT ='+str(h))

            nf = self.nf
            h0 = slim.fully_connected(h, 7*7*nf, activation_fn=None) # h0
            net = tf.reshape(h0, [-1, 7, 7, nf])

            print('Decoder layer ='+str(net))
            with slim.arg_scope([slim.conv2d], kernel_size=[3,3], padding='SAME', activation_fn=tf.nn.elu):

                print('Decoder layer ='+str(net))
                #net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)
                net = tf.image.resize_nearest_neighbor(net, [14, 14]) # upsampling

                print('Decoder layer ='+str(net))
                #net = slim.conv2d(net, nf)
                net = slim.conv2d(net, 2*nf)
                net = tf.image.resize_nearest_neighbor(net, [28, 28])
                print('Decoder layer ='+str(net))
                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)

                print('Decoder layer ='+str(net))
                net = slim.conv2d(net, 1, activation_fn=None)
            print('***Decoder OUTPUT ='+str(net))
            return net

    def _discriminator(self, X, reuse=False, name='discriminator'):
        print('############### DISCRIMINATOR START ##########################')

        with tf.variable_scope('D', reuse=reuse):

            h = self._encoder(X, reuse=reuse)
            x_recon = self._decoder(h, reuse=reuse)

            energy_dense = tf.abs(X-x_recon) # L1 loss
            energy = tf.reduce_mean(energy_dense, name=name)
            #energy=tf.reduce_mean(tf.reduce_sum((X - x_recon)**2, 1))
        print('############### DISCRIMINATOR END ##########################')

        return energy, h

    def _generator(self, z, reuse=False):
        print('############### GENERATOR START##########################')
        print('Input tensor='+str(z))
        with tf.variable_scope('G', reuse=reuse):
            x_fake = self._decoder(z, reuse=reuse)

            ''' simple net addition '''
            #x_fake = tf.nn.sigmoid(x_fake)
        print('############### GENERATOR END ##########################')

        return x_fake


def model(data,
            hparams,
            mode):

    #allocate a BEGAN class instance and return the target model:
    return Began(data,
                hparams.generatorCodeSize,
                hparams).get_model_graph(mode)
