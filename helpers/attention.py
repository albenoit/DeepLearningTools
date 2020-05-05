''' a collection of attention modules '''

import tensorflow as tf
import helpers.loss as loss

def squeeze_excitation(input_features):
    ''' apply squeeze and exitation auto attention
        from https://arxiv.org/abs/1709.01507
        extended to support 2D and 3D data
        Args:
            input_features, the input features of shape [batch, height, width, channels]
        Returns:
            the channel weighed features weights of same shape as the input

    '''
    with tf.name_scope('Squeeze_and_excitation'):
        #reduction_ratio=16 #default reduction factor used to reduce the dimension of the first channel interractions
        #first apply gobal average pooling to obtain channel average activation
        print('input features shape', input_features.shape)
        features_dim=len(input_features.get_shape().as_list())
        if features_dim==4:
            print('2D squeeze and axcitation module')
            features_averages= tf.keras.layers.GlobalAveragePooling2D()(input_features)
        elif features_dim==5:
            print('3D squeeze and axcitation module')
            features_averages= tf.keras.layers.GlobalAveragePooling3D()(input_features)
        else:
            raise ValueError('Input features are expected to be 4D(2D data) or 5D(3D, volumetric data)')

        print('pooled features shape', features_averages.shape)
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

        
def spatial_attention(input_features):
    ''' create a spatial maps that highlights regions of interest (autoattention based..)
    Args:
        input_features, the input features of shape [batch, height, width, channels]
    Returns:
        weights of shape [batch, height, width, 1]

    from https://arxiv.org/pdf/2001.07645.pdf
    '''
    with tf.name_scope('Spatial_attention'):
        print('input features shape', input_features.shape)
        features_dim=len(input_features.get_shape().as_list())
        if features_dim==4:
            print('2D spatial attention module')
            conv_op=tf.keras.layers.Conv2D
            kernel=[1,1]
            strides=(1,1)
            dilation_rate=(1, 1)
        elif features_dim==5:
            print('3D attention module')
            conv_op=tf.keras.layers.Conv3D
            kernel=(1,1,1)
            dilation_rate=(1, 1, 1)
        else:
            raise ValueError('Input features are expected to be 4D(2D data) or 5D(3D, volumetric data)')
        print('input features shape', input_features.shape)
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


def dual_attention(input_features):
    ''' combine spatial AND channel attention weighting (autoattention based..)
    Args:
        4D or 5D data if respecively image or volume samples
        input_features, the input features of shape [batch, height, width, (depth), channels]
    Returns:
        spatially and channely weighted features of same shape as input [batch, height, width, (depth), channels]

    from https://arxiv.org/pdf/2001.07645.pdf
    '''
    return squeeze_excitation(input_features)*(1.+spatial_attention(input_features))