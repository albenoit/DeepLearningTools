""" Densely connected network model
inspired from https://github.com/LaurentMazare/deep-models/blob/master/densenet/densenet.py
and and https://github.com/SimJeg/FC-DenseNet/blob/master/FC-DenseNet.py
@author Alexandre Benoit, LISTIC
"""
import tensorflow as tf
import numpy as np

def weight_variable(shape):
    '''MSRA initialization of a given weigths tensor
    @param shape, the 4d tensor shape
    variable is allocated on the CPU memory even if processing will use it on GPU
    '''
    n= np.prod(shape[:3])#n_input_channels*kernelShape
    trunc_stddev = np.sqrt(1.3 * 2.0 / n)
    initial = tf.truncated_normal(shape, 0.0, trunc_stddev)
    weights=tf.get_variable(name='weights', initializer=initial)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
    return weights

def bias_variable(shape):
    ''' basic constant bias variable init (a little above 0)
    @param shape, the 4d tensor shape
    variable is allocated on the CPU memory even if processing will use it on GPU
    '''
    initial = tf.constant(0.01, shape=shape)
    return tf.get_variable(name='biases', initializer=initial)

"""
def conv2d(input_features, outing_nb_features, kernel_size, with_bias=True):
    with tf.variable_scope('conv2d'):
        W = weight_variable([ kernel_size, kernel_size, input_features.get_shape().as_list()[-1], outing_nb_features ])
        conv = tf.nn.conv2d(input_features, W, [ 1, 1, 1, 1 ], padding='SAME')
        if with_bias:
            conv= conv + bias_variable([ outing_nb_features ])
        return conv
"""
def conv3d(input_features, outing_nb_features, kernel_size, with_bias=True, squeeze=False, strides=[1,1,1]):
    with tf.variable_scope('conv3d'):
        if squeeze is False:#normal 3D conv behavior
            W = weight_variable([ kernel_size, kernel_size, kernel_size, input_features.get_shape().as_list()[-1], outing_nb_features ])
            conv = tf.nn.conv3d(input_features, W, [1]+strides+[1], padding='SAME')
        else:
            W = weight_variable([ kernel_size, kernel_size, input_features.get_shape().as_list()[-2], input_features.get_shape().as_list()[-1], outing_nb_features ])
            conv = tf.nn.conv3d(input_features, W, [1]+strides+[1], padding='VALID')
            print('SQUEEZING the 3rd dimension, filter shape='+str(W.get_shape().as_list()))
            #squeezing the 4th dimension
            conv=tf.squeeze(conv, [3], "squeeze_3rd_dim")
        if with_bias:
            conv= conv + bias_variable([ outing_nb_features ])

        print('DEBUG CONV3D, conv (input, output) shapes='+str((input_features.get_shape().as_list(), conv.get_shape().as_list())))
        return conv

"""
def conv2d_upscale(input_features, kernel_size, with_bias=True):
    with tf.variable_scope('conv2d_upscale'):
        print('conv2d_upscale: input feature shape='+str(input_features.get_shape().as_list()))
        nb_channels=input_features.get_shape().as_list()[-1]
        W = weight_variable([ kernel_size, kernel_size, nb_channels, nb_channels ])
        output_shape=[input_features.get_shape().as_list()[0]]+(np.array(input_features.get_shape().as_list()[1:3])*2).tolist()+[nb_channels]
        print('upscaling weights shape='+str(W.get_shape().as_list()))
        print('expected output shape='+str())
        conv = tf.nn.conv2d_transpose(input_features, W,
                                        output_shape=output_shape,
                                        strides=[ 1, 2, 2, 1 ],
                                        padding='SAME')
        if with_bias:
          conv=conv + bias_variable([ nb_channels ])

        #upscale to initial resolution
        print('conv2d_upscale: output feature shape='+str(conv.get_shape().as_list()))

        #upscaled_conv=tf.image.resize_bilinear(images=conv, size=(np.array(conv.get_shape()[1:3].as_list())*2).tolist(), align_corners=True, name='output_segmentation_upsampled')
        return conv
"""
def conv3d_upscale(input_features, kernel_size, with_bias=True):
    with tf.variable_scope('conv3d_upscale'):
        print('conv3d_upscale: input feature shape='+str(input_features.get_shape().as_list()))
        nb_channels=input_features.get_shape().as_list()[-1]
        W = weight_variable([ kernel_size, kernel_size, kernel_size, nb_channels, nb_channels ])
        output_shape=[input_features.get_shape().as_list()[0]]+(np.array(input_features.get_shape().as_list()[1:4])*2).tolist()+[nb_channels]
        print('upscaling weights shape='+str(W.get_shape().as_list()))
        print('expected output shape='+str(output_shape))
        conv = tf.nn.conv3d_transpose(input_features, W,
                                        output_shape=output_shape,
                                        strides=[ 1, 2, 2, 2, 1 ],
                                        padding='SAME')
        if with_bias:
          conv=conv + bias_variable([ nb_channels ])

        #upscale to initial resolution
        print('conv3d_upscale: output feature shape='+str(conv.get_shape().as_list()))

        #upscaled_conv=tf.image.resize_bilinear(images=conv, size=(np.array(conv.get_shape()[1:3].as_list())*2).tolist(), align_corners=True, name='output_segmentation_upsampled')
        return conv


"""
def composite_function(input_features, out_features, kernel_size, is_training, keep_prob, name):
    '''Motivated by [12], we define H()
    as a composite function of three consecutive operations:
    batch normalization (BN) [14], followed by a rectified linear
    unit (ReLU) [6] and a 3 * 3 convolution (Conv).
    '''
    with tf.variable_scope('Composite_'+str(name)):
        preprocessed_features = tf.nn.relu(tf.contrib.layers.layer_norm(input_features))#, training=is_training))
        new_features = conv2d(preprocessed_features, out_features, kernel_size)
        new_features = tf.nn.dropout(new_features, keep_prob)
        return new_features
"""
def composite_function_3d(input_features, out_features, kernel_size, is_training, keep_prob, name):
    '''Motivated by [12], we define H()
    as a composite function of three consecutive operations:
    batch normalization (BN) [14], followed by a rectified linear
    unit (ReLU) [6] and a 3 * 3 convolution (Conv).
    '''
    with tf.variable_scope('Composite_3d_'+str(name)):
        preprocessed_features = tf.nn.relu(tf.contrib.layers.layer_norm(input_features))#, training=is_training))
        new_features = conv3d(preprocessed_features, out_features, kernel_size)
        new_features = tf.nn.dropout(new_features, keep_prob)
        return new_features

"""
def transition_up(input_features, bloc_idx):

    with tf.variable_scope('TransitionUp_'+str(bloc_idx)):
        #upscale encoded features
        transition=conv2d_upscale(input_features, 3)
        print('Transition up, upscaling encoded features {inshape}'.format(inshape=input_features.get_shape().as_list()))
        return transition
"""
def transition_up_3d(input_features, bloc_idx):

    with tf.variable_scope('TransitionUp_3d_'+str(bloc_idx)):
        #upscale encoded features
        transition=conv3d_upscale(input_features, 3)
        print('Transition up, upscaling encoded features {inshape}'.format(inshape=input_features.get_shape().as_list()))
        return transition

"""
def transition_down(input_features, is_training, keep_prob, block_idx):
    '''The transition layers
    used in our experiments consist of a batch normalization
    layer and an 1*1 convolutional layer followed by a 2*2 average
    pooling layer. Actually, a ReLU is used, keeping the BN-ReLU-Conv structure
    ===> TODO, check if pooling could be replaced by learned convolutions (help reconstruct semantic map?)
    ------> would be conv2d with kernel=3 and strite=2 ?
    '''
    with tf.variable_scope('TransitionDown_'+str(block_idx)):
        preprocessed_features = tf.nn.relu(tf.contrib.layers.layer_norm(input_features))#, training=is_training))
        new_features = conv2d(preprocessed_features, preprocessed_features.get_shape().as_list()[-1], kernel_size=1)
        new_features = tf.nn.dropout(new_features, keep_prob)
        transition = tf.nn.max_pool(new_features, ksize=[ 1, 2, 2, 1 ], strides=[1, 2, 2, 1 ], padding='SAME')
        return transition
"""
def transition_down_3d(input_features, is_training, keep_prob, block_idx, strided_conv=False):
    '''The transition layers
    used in our experiments consist of a batch normalization
    layer and an 1*1 convolutional layer followed by a 2*2 average
    pooling layer. Actually, a ReLU is used, keeping the BN-ReLU-Conv structure
    ===> TODO, check if pooling could be replaced by learned convolutions (help reconstruct semantic map?)
    ------> would be conv2d with kernel=3 and strite=2 ?
    '''
    with tf.variable_scope('TransitionDown_3d_'+str(block_idx)):
        preprocessed_features = tf.nn.relu(tf.contrib.layers.layer_norm(input_features))#, training=is_training))
        if strided_conv:
            transition = conv3d(preprocessed_features, preprocessed_features.get_shape().as_list()[-1], kernel_size=3, strides=[2,2,2])
        else:
            new_features = conv3d(preprocessed_features, preprocessed_features.get_shape().as_list()[-1], kernel_size=1)
            new_features = tf.nn.dropout(new_features, keep_prob)
            transition = tf.nn.max_pool3d(new_features, ksize=[ 1, 2, 2, 2, 1 ], strides=[1, 2, 2, 2, 1 ], padding='SAME')
        return transition

"""
def block(input, layers, growth, is_training, keep_prob, blockID):
    '''each layer of the block receives all the preceeding data
    but the block output feature maps do not include the initial input features
    '''
    with tf.variable_scope('DenseBlock_'+str(blockID)):
        feature_maps = input
        block_feature_maps=[]
        print('NEW BLOCK:')
        for idx in xrange(layers):
            print('--> new layer : applying layer with {growth} neurons on input features of shape {inshape}'.format(growth=growth, inshape=feature_maps.get_shape().as_list()))
            new_feature_maps = composite_function(feature_maps, growth, 3, is_training, keep_prob, name=idx)
            if idx<(layers-1):
                feature_maps=tf.concat([feature_maps, new_feature_maps], axis=3, name='block_intra_features_concat')
            block_feature_maps.append(new_feature_maps)
        if len(block_feature_maps)>1:
            block_out=tf.concat(block_feature_maps, axis=3, name='block_layers_concat')
        else:
            block_out=new_feature_maps
        print('==> block output shape = {outshape}'.format(outshape=block_out.get_shape().as_list()))
    return block_out
"""
def block_3d(input, layers, growth, is_training, keep_prob, blockID, dense_block=True):
    '''each layer of the block receives all the preceeding data
    but the block output feature maps do not include the initial input features
    '''
    with tf.variable_scope('DenseBlock_3d_'+str(blockID)):
        feature_maps = input
        block_feature_maps=[]
        print('NEW BLOCK:')
        for idx in xrange(layers):
            print('--> new layer : applying layer with {growth} neurons on input features of shape {inshape}'.format(growth=growth, inshape=feature_maps.get_shape().as_list()))
            new_feature_maps = composite_function_3d(feature_maps, growth, 3, is_training, keep_prob, name=idx)
            if idx<(layers-1) and dense_block is True:
                feature_maps=tf.concat([feature_maps, new_feature_maps], axis=4, name='block_intra_features_concat')
            else:
                feature_maps=new_feature_maps
            block_feature_maps.append(new_feature_maps)
        if len(block_feature_maps)>1  and dense_block is True:
            block_out=tf.concat(block_feature_maps, axis=4, name='block_layers_concat')
        else:
            block_out=new_feature_maps
        print('==> block output shape = {outshape}'.format(outshape=block_out.get_shape().as_list()))
    return block_out

def avg_pool(input, s):
  return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

def model(  data,
            hparams,
            mode
            ):
    from tensorflow.contrib.learn import ModeKeys
    is_training = mode == ModeKeys.TRAIN

    #user config
    dropout_rate=0.2
    keep_prob=1.0-dropout_rate
    print(hparams.debug)
    #basic architexture for testing purpose
    nb_layers_sequence_encoding=[2, 2, 2]
    bottleneck_nb_layers=1
    growth_rate=16
    output_only_inputs_last_decoding_block=False
    use_dense_block=hparams.denseBlocks #if False, then the architecture will not include dense connections and will resemble UNet
    use_skip_connections=hparams.skipConnections #set True to activate skip conectionx between the encoder and decoder
    variationnal_AE=hparams.isVAE#set True to transform the AE into a VAE
    use_stridedconv_insteadof_maxpool=False
    '''
    #FC-DenseNet-103 architecture:
    nb_layers_sequence_encoding=[4, 5, 7, 10, 12]
    bottleneck_nb_layers=15
    growth_rate=16
    #set True in order to avoid the n-1 decoding block to be connected to the final conv outputs (False by default)
    output_only_inputs_last_decoding_block=False
    '''
    number_of_encoding_blocks=len(nb_layers_sequence_encoding)
    n_layers_per_block=nb_layers_sequence_encoding+[bottleneck_nb_layers]+nb_layers_sequence_encoding[::-1]
    print('Using {blocks} blocks for each encoding and decoding branch with the following number of layers: {layers_per_blocks}'.format(blocks=number_of_encoding_blocks,
                                                                                                                                layers_per_blocks=n_layers_per_block))
    total_nb_layers=np.sum(n_layers_per_block)+len(n_layers_per_block)+1
    print('Expected number of layers (including input, output and transition blocks)= '+str(total_nb_layers))
    print('RAW data shape='+str(data.get_shape().as_list()))

    skip_connection_list = []
    field_of_view=1
    with tf.variable_scope('Encoder'):
        '''with tf.variable_scope('Input'):
            input_block_features_count=48
            first_filters_size=20
            feature_maps = conv3d(tf.expand_dims(data, -1), input_block_features_count, first_filters_size)
            #feature_maps=tf.nn.max_pool(input_features, [ 1, 3, 3, 1 ], strides=[1, 2, 2, 1 ], padding='SAME')
            print('Input block output shape='+str(feature_maps.get_shape().as_list()))
            #field_of_view+=(first_filters_size-1)
        '''
        feature_maps=data
        for blockID in xrange(number_of_encoding_blocks):
            feature_maps_block = block_3d(feature_maps, n_layers_per_block[blockID], growth_rate, is_training, keep_prob, blockID, use_dense_block)
            #concatenate with the input features
            if use_dense_block is True:
                feature_maps_block=tf.concat([feature_maps_block, feature_maps], axis=4, name='encoding_block_layers_concat_'+str(blockID))
            # At the end of the dense block, the current stack is stored in the skip_connections list
            skip_connection_list.append(feature_maps_block)
            feature_maps=transition_down_3d( feature_maps_block, is_training, keep_prob, blockID, use_stridedconv_insteadof_maxpool)
            print('** concat(Block+input)+transition down shape='+str(feature_maps.get_shape().as_list()))
            #update field of view
            field_of_view+=n_layers_per_block[blockID]*2
        #reverse skip layers list to apply it in the decding layers set
        skip_connection_list = skip_connection_list[::-1]

        #central bottleneck
    with tf.variable_scope('Bottleneck'):
        print('Central bottleneck with {central_nb_layers} layers'.format(central_nb_layers=n_layers_per_block[number_of_encoding_blocks]))
        last_encoding_feature_maps = block_3d(feature_maps, n_layers_per_block[number_of_encoding_blocks], growth_rate, is_training, keep_prob, number_of_encoding_blocks, use_dense_block)
        field_of_view+=n_layers_per_block[number_of_encoding_blocks]*2
        last_encoding_feature_maps_shape=last_encoding_feature_maps.get_shape().as_list()
        if hparams.debug_sess:
          last_encoding_feature_maps=tf.check_numerics(last_encoding_feature_maps, message='Encoder : NaN in last_encoding_feature_maps')
        #add specific layers for variationnal autoencoding
        if variationnal_AE: #hints from https://github.com/vividda/3D-PhysNet/blob/master/main-3D-PhysNet-IJCAI.py
          with tf.name_scope('variationnal_encoding'):

            with tf.name_scope('3D_to_1D_flattening'):
                #activation+nprmalization bbefore feeding into the next layers
                vae_code_=tf.nn.relu(tf.contrib.layers.layer_norm(last_encoding_feature_maps))
                encoder_conv_out_flat=tf.layers.flatten(vae_code_)
                print('Bottleneck: Code size before variationnal encoding='+str(encoder_conv_out_flat))
                encoder_vect = tf.layers.dense(inputs=encoder_conv_out_flat,
                                             units=500,#5000,
                                             activation=tf.nn.relu,
                                             use_bias=True,
                                             kernel_initializer= tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution="normal"), #MSRA init
                                             name='flatten_code')
            with tf.name_scope('Mean_and_std'):
                z_mean = tf.layers.dense(inputs=encoder_conv_out_flat,
                                         units=32,#800,
                                         activation=None,
                                         use_bias=True,
                                         kernel_initializer= tf.glorot_uniform_initializer(),#Xavier init
                                         name='z_mean'
                                         )
                                         #tools.Ops.fc(lfc, out_d=800, name='z_mean')
                z_std = tf.layers.dense(inputs=encoder_conv_out_flat,
                                         units=32,#800,
                                         activation=None,
                                         use_bias=True,
                                         kernel_initializer= tf.glorot_uniform_initializer(),#Xavier init
                                         name='z_std'
                                         )
                                         #z_std = tools.Ops.fc(lfc, out_d=800, name='z_std')
                if hparams.debug_sess:
                  z_mean=tf.check_numerics(z_mean, message='Encoder : NaN in z_mean')
                  z_std=tf.check_numerics(z_std, message='Encoder : NaN in z_std')

            with tf.name_scope('Gaussian_distribution_sampling'):
                # Sampler: Normal (gaussian) random distribution
                eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                                       name='epsilon')
                samples = z_mean + tf.exp(z_std / 2.0) * eps

                #if willing to introduce some external data in the code, to this here:
                #lfc = tf.concat([lfc, C_G], axis=1)

            with tf.name_scope('sampled_code'):
                last_encoding_feature_maps = tf.layers.dense(inputs=samples,
                                         units=np.prod(last_encoding_feature_maps_shape[1:]),
                                         activation=tf.nn.relu,
                                         use_bias=True,
                                         kernel_initializer= tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution="normal"), #MSRA init
                                         name='sampled_code'
                                         )
                print('decoder_first_dense_layer:'+str())

                                         #lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=np.prod(last_encoding_feature_maps_shape[1:]), name='fc2'),
                                         #name='relu')
                last_encoding_feature_maps = tf.reshape(last_encoding_feature_maps, last_encoding_feature_maps_shape)
    """
    #image classification task branch
    with tf.variable_scope('Classifier'):
        features_average = avg_pool(last_encoding_feature_maps, last_encoding_feature_maps.get_shape().as_list()[1])
        final_dim = last_encoding_feature_maps.get_shape().as_list()[-1]
        features_average_flat = tf.reshape(features_average, [ -1, final_dim ])
        Wfc = weight_variable([ final_dim, n_outputs ])
        bfc = bias_variable([ n_outputs ])
        logits_classif= tf.matmul(features_average_flat, Wfc) + bfc
    """
    if hparams.debug_sess:
      last_encoding_feature_maps=tf.check_numerics(last_encoding_feature_maps, message='Decoder : NaN in last_encoding_feature_maps')

    with tf.variable_scope('Reconstruction'):
        # We store now the output of the next dense block in a list. We will only upsample these new feature maps
        block_to_upsample = []
        decoding_feature_maps=last_encoding_feature_maps
        for blockID in xrange(number_of_encoding_blocks):
            nlayers=n_layers_per_block[number_of_encoding_blocks + blockID+1]
            print('creating a block with {nlayers} layers'.format(nlayers=nlayers))

            # Transition Up ( Upsampling )
            feature_maps_up = transition_up_3d(decoding_feature_maps, bloc_idx= blockID)

            #concatenate with the skip connection)
            print('Transition up, concatenated upscaled encoded features {upshape} with skip layer {skipshape}'.format(upshape=feature_maps_up.get_shape().as_list(),
                                                                                                                skipshape=skip_connection_list[blockID].get_shape().as_list()))
            if use_skip_connections is True:
                feature_maps_in=tf.concat([feature_maps_up, skip_connection_list[blockID]], axis=4, name='decoding_block_layers_concat_'+str(blockID))
            else:
                feature_maps_in=feature_maps_up
            print('Transition up+skip layers output features {outshape}'.format(outshape=feature_maps_in.get_shape().as_list()))
            # apply dense block
            decoding_feature_maps = block_3d(feature_maps_in, nlayers, growth_rate, is_training, keep_prob, blockID, use_dense_block)

        with tf.variable_scope('reconstruction_output'):
            print('last decoding feature map='+str(decoding_feature_maps))
            reconstructed_channels_output=data.get_shape().as_list()[-1]
            if output_only_inputs_last_decoding_block is True:
                logits_semantic = conv3d(decoding_feature_maps, reconstructed_channels_output, kernel_size=1, squeeze=True)
            else:
                logits_semantic = conv3d(tf.concat([decoding_feature_maps, feature_maps_in], axis=4), reconstructed_channels_output, kernel_size=1, squeeze=False)

            logits_semantic =tf.sigmoid(logits_semantic, name='output_sigmoid')
    print('logits_semantic shape='+str(logits_semantic.get_shape().as_list()))
    print('Net global image classificer out shape = '+str(logits_semantic.get_shape().as_list()))

    print('Semantic segmentation pixel field of view='+str(field_of_view))

    output_dict={'code':last_encoding_feature_maps, 'reconstructed_data':logits_semantic}
    print('Model output dict=',output_dict)

    #add optionnal outputs
    if variationnal_AE:
        output_dict['encoder_vect']=encoder_conv_out_flat
        output_dict['z_mean']=z_mean
        output_dict['z_std']=z_std
        output_dict['z_mean_std']=tf.concat([z_mean, z_std], axis=1)


    #add each skip connexion output for embedding
    all_skips_list=[]
    all_skips_list_flat=[]
    for id, skip_layer in enumerate(skip_connection_list):
        output_dict['skip_'+str(id)]=skip_layer
        all_skips_list_flat.append(tf.layers.flatten(skip_layer))
        '''#finally concatenate all the skip lapeyrs
        all_skips_list.append(tf.image.resize_nearest_neighbor(
                                               skip_layer,
                                               logits_semantic.get_shape().as_list()[1:4],
                                               align_corners=True,
                                              )
                        )
        '''
    #all_skips=tf.concat(all_skips_list, axis=-1)
    #output_dict['all_skips']=all_skips
    #concat code+skiplayers
    output_dict['code_with_all_skips']=tf.concat([tf.layers.flatten(last_encoding_feature_maps)]+all_skips_list_flat, axis=1)

    return output_dict
