# ========================================
# FileName: attention.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A collection of CNN attention modules
# for DeepLearningTools.
# =========================================

import tensorflow as tf
from deeplearningtools.helpers import loss
from deeplearningtools.helpers import model

def squeeze_excitation(input_features: tf.Tensor) -> tf.Tensor:
    """
    Apply squeeze and excitation auto attention.
    
    This function implements the squeeze and excitation mechanism, as described in the paper
    "Squeeze-and-Excitation Networks" (https://arxiv.org/abs/1709.01507).
    It extended supports 2D and 3D data as input.
    
    :param input_features: The input features of shape [batch, height, width, channels] or [batch, depth, height, width, channels].
    :type input_features: tf.Tensor
        
    :return: The channel-weighted features of the same shape as the input.
    :rtype: tf.Tensor
    """
    with tf.name_scope('Squeeze_and_excitation'):
        #reduction_ratio=16 #default reduction factor used to reduce the dimension of the first channel interractions
        #first apply gobal average pooling to obtain channel average activation
        #print('input features shape', input_features.shape)
        features_dim=len(input_features.get_shape().as_list())
        if features_dim==4:
            #print('2D squeeze and axcitation module')
            features_averages= tf.keras.layers.GlobalAveragePooling2D()(input_features)
        elif features_dim==5:
            #print('3D squeeze and axcitation module')
            features_averages= tf.keras.layers.GlobalAveragePooling3D()(input_features)
        else:
            raise ValueError('Input features are expected to be 4D(2D data) or 5D(3D, volumetric data)')

        #print('pooled features shape', features_averages.shape)
        #excitation model
        first_interractions=tf.keras.layers.Dense(units=input_features.shape[-1]//16,
                                activation=tf.keras.activations.relu,
                                kernel_initializer=tf.keras.initializers.Orthogonal(),
                                bias_initializer=tf.initializers.constant(0.001),
                                kernel_regularizer=loss.Regularizer_soft_orthogonality(),
                                )(features_averages)
        channel_weights=tf.keras.layers.Dense(units=input_features.shape[-1],
                                activation=tf.keras.activations.sigmoid,
                                kernel_initializer=tf.keras.initializers.Orthogonal(),
                                bias_initializer=tf.initializers.constant(0.001),
                                kernel_regularizer=loss.Regularizer_soft_orthogonality(),
                                )(first_interractions)
        #return the weighted features
        channel_weights = tf.reshape(channel_weights, [-1]+[1]*(features_dim-2)+[input_features.shape[-1]])
        
        return input_features * channel_weights

        
def spatial_attention(input_features: tf.Tensor) -> tf.Tensor:
    """
    Create a spatial map that highlights regions of interest (auto-attention based).

    This function implements the spatial attention mechanism, as described in the paper
    "Spatial Attention in Convolutional Networks for Image Captioning" (https://arxiv.org/pdf/2001.07645.pdf).
    It supports 2D and 3D data as input.

    :param input_features: The input features of shape [batch, height, width, channels] or [batch, depth, height, width, channels].
    :type input_features: tf.Tensor

    :return: The weights of shape [batch, height, width, 1].
    :rtype: tf.Tensor
    """
    with tf.name_scope('Spatial_attention'):
        #print('input features shape', input_features.shape)
        features_dim=len(input_features.get_shape().as_list())
        if features_dim==4:
            #print('2D spatial attention module')
            conv_op=tf.keras.layers.Conv2D
            kernel=[1,1]
            strides=(1,1)
            dilation_rate=(1, 1)
        elif features_dim==5:
            #print('3D attention module')
            conv_op=tf.keras.layers.Conv3D
            kernel=(1,1,1)
            dilation_rate=(1, 1, 1)
        else:
            raise ValueError('Input features are expected to be 4D(2D data) or 5D(3D, volumetric data)')
        #print('input features shape', input_features.shape)
        #excitation model
        first_interractions=conv_op(
                        filters=input_features.shape[-1]//2,
                        kernel_size=kernel,
                        strides=strides,
                        padding='same',
                        data_format='channels_last',
                        dilation_rate=dilation_rate,
                        activation=tf.keras.activations.relu,
                        use_bias=True,
                        kernel_initializer=tf.keras.initializers.Orthogonal(),
                        bias_initializer=tf.initializers.constant(0.001),
                        kernel_regularizer=loss.Regularizer_soft_orthogonality(),
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None
                       )(input_features)
        spatial_weights=conv_op(
                        filters=1,
                        kernel_size=kernel,
                        strides=strides,
                        padding='same',
                        data_format='channels_last',
                        dilation_rate=dilation_rate,
                        activation=tf.keras.activations.sigmoid,
                        use_bias=True,
                        kernel_initializer=tf.keras.initializers.Orthogonal(),
                        bias_initializer=tf.initializers.constant(0.001),
                        kernel_regularizer=loss.Regularizer_soft_orthogonality(),
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None
                       )(first_interractions)
        #return the weighted features
        return spatial_weights


def dual_attention(input_features: tf.Tensor) -> tf.Tensor:
    """
    Combine spatial and channel attention weighting (auto-attention based).

    This function combines the spatial and channel attention mechanisms, as described in the paper 
    "Spatial Attention in Convolutional Networks for Image Captioning" (https://arxiv.org/pdf/2001.07645.pdf).
    It takes 4D or 5D data as input, depending on whether it's image or volume samples.

    :param input_features: The input features of shape [batch, height, width, (depth), channels].
    :type input_features: tf.Tensor

    :return: The spatially and channel-wise weighted features of the same shape as the input [batch, height, width, (depth), channels].
    :rtype: tf.Tensor
    """
    return squeeze_excitation(input_features)*(1.+spatial_attention(input_features))


def aconv(input_features: tf.Tensor, sampling_rate: int = 8) -> tf.Tensor:
    """
    Combine spatial and channel attention weighting.

    This function combines spatial and channel attention weighting using the aconv mechanism.
    It supports 4D and 5D data as input, respectively image or volume samples.
    Actually very similar to squeeze and excitation BUT squeeze is only a subsampling thus forcing local neighbors agregation.
    The excitation process then relies on convolutions before the final upsampling and activation step.

    :param input_features: The input features of shape [batch, height, width, channels] or [batch, depth, height, width, channels].
    :type input_features: tf.Tensor

    :param sampling_rate: The downsampling rate applied to the input features, defaults to 8.
    :type sampling_rate: int, optional

    :return: The spatially and channel-weighted features of the same shape as the input [batch, height, width, (depth), channels].
    :rtype: tf.Tensor
    """
    features_dim=len(input_features.get_shape().as_list())
    features_subsampler=None
    features_upscaler=None
    with tf.name_scope('attention_convolution'):
        if features_dim==4:
            #print('2D spatial attention module')
            conv_op=tf.keras.layers.Conv2D
            features_subsampler=tf.keras.layers.AveragePooling2D
            features_upscaler=tf.keras.layers.UpSampling2D
            sampler_rate=(sampling_rate, sampling_rate)
            kernel=[3,3]
            strides=(1,1)
            dilation_rate=(1, 1)
        elif features_dim==5:
            #print('3D attention module')
            conv_op=tf.keras.layers.Conv3D
            features_subsampler=tf.keras.layers.AveragePooling3D
            features_upscaler=tf.keras.layers.UpSampling3D
            sampler_rate=(sampling_rate, sampling_rate, sampling_rate)
            kernel=(3, 3, 3)
            strides=(1, 1, 1)
            dilation_rate=(1, 1, 1)
        else:
            raise ValueError('Input features are expected to be 4D(2D data) or 5D(3D, volumetric data)')

        #Attention convolution operator definition
        #print('Attention convolution module')
        
        #print('input features',input_features)
        #squeeze
        features_subsampled=features_subsampler(pool_size=sampler_rate,
                                                strides=None,
                                                padding='valid',
                                                data_format=None)(input_features)
        #print('subsampled features',features_subsampled)

        #Excite
        first_interractions=conv_op( #CVPR paper considers 2 groups of convolution
                            filters=input_features.shape[-1]//4,#CVPR paper : empirically reduce by a factor of 4
                            kernel_size=kernel,
                            strides=strides,
                            padding='same',
                            data_format='channels_last',
                            dilation_rate=dilation_rate,
                            activation=tf.keras.activations.relu,
                            use_bias=True,
                            kernel_initializer=tf.keras.initializers.Orthogonal(),
                            bias_initializer=tf.initializers.constant(0.001),
                            kernel_regularizer=loss.Regularizer_soft_orthogonality(),
                            bias_regularizer=None,
                            activity_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None
                        )(features_subsampled)
        weights=conv_op(
                            filters=1,
                            kernel_size=kernel,
                            strides=strides,
                            padding='same',
                            data_format='channels_last',
                            dilation_rate=dilation_rate,
                            activation=tf.keras.activations.sigmoid,
                            use_bias=True,
                            kernel_initializer=tf.keras.initializers.Orthogonal(),
                            bias_initializer=tf.initializers.constant(0.001),
                            kernel_regularizer=loss.Regularizer_soft_orthogonality(),
                            bias_regularizer=None,
                            activity_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None
                        )(first_interractions)
        #Finally upsample
        upscaled_weights=features_upscaler(size=sampler_rate, data_format=None, interpolation='nearest')(weights)
        #return the weighted features
        return input_features * upscaled_weights


def aconv2(input_features: tf.Tensor, sampling_rate: int = 8) -> tf.Tensor:
    """
    Combine spatial and channel attention weighting.

    This function combines spatial and channel attention weighting using the aconv2 mechanism.
    It supports 2D and 3D data as input.

    :param input_features: The input features of shape [batch, height, width, channels] or [batch, height, width, (depth), channels].
    :type input_features: tf.Tensor

    :param sampling_rate: The downsampling rate applied to the input features, defaults to 8.
    :type sampling_rate: int, optional

    :return: The spatially and channel-weighted features of the same shape as the input [batch, height, width, (depth), channels].
    :rtype: tf.Tensor

    References:REVISED VERSION of "Squeeze-and-Attention Networks for Semantic Segmentation" (https://openaccess.thecvf.com/content_CVPR_2020/html/Zhong_Squeeze-and-Attention_Networks_for_Semantic_Segmentation_CVPR_2020_paper.html)
    """
    features_dim=len(input_features.get_shape().as_list())
    features_subsampler=None
    with tf.name_scope('attention_convolution'):
        if features_dim==4:
            #print('2D spatial attention module')
            conv_op=tf.keras.layers.Conv2D
            features_subsampler=tf.keras.layers.AveragePooling2D
            sampler_rate=(sampling_rate, sampling_rate)
            kernel=[3, 3]
            strides=(1, 1)
            dilation_rate=(1, 1)
        elif features_dim==5:
            #print('3D attention module')
            conv_op=tf.keras.layers.Conv3D
            features_subsampler=tf.keras.layers.AveragePooling3D
            sampler_rate=(sampling_rate, sampling_rate, sampling_rate)
            kernel=(3, 3, 3)
            strides=(1, 1, 1)
            dilation_rate=(1, 1, 1)
        else:
            raise ValueError('Input features are expected to be 4D(2D data) or 5D(3D, volumetric data)')
        
        #Attention convolution operator definition
        #print('REVISED attention convolution module')

        #print('input features',input_features)
        #squeeze
        features_subsampled=features_subsampler(pool_size=sampler_rate,
                                                strides=None,
                                                padding='valid',
                                                data_format=None)(input_features)
        #print('subsampled features',features_subsampled)

        #Excite
        first_interractions=conv_op( #CVPR paper considers 2 groups of convolution
                            filters=input_features.shape[-1],#//4,#CVPR paper : empirically reduce by a factor of 4
                            kernel_size=kernel,
                            strides=strides,
                            padding='same',
                            data_format='channels_last',
                            dilation_rate=dilation_rate,
                            activation=tf.keras.activations.relu,
                            use_bias=True,
                            kernel_initializer=tf.keras.initializers.Orthogonal(),
                            bias_initializer=tf.initializers.constant(0.001),
                            kernel_regularizer=loss.Regularizer_soft_orthogonality(),
                            bias_regularizer=None,
                            activity_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None
                        )(features_subsampled)
        upscaled_weights=model.SubpixelConv2D(first_interractions.get_shape().as_list(), factor=sampling_rate)(first_interractions)
        #return the weighted features
        return input_features * upscaled_weights


