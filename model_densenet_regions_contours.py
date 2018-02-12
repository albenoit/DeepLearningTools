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
    with tf.device('/cpu:0'):
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
    with tf.device('/cpu:0'):
        initial = tf.constant(0.01, shape=shape)
        return tf.get_variable(name='biases', initializer=initial)

def conv2d(input_features, outing_nb_features, kernel_size, with_bias=True):
    with tf.variable_scope('conv2d'):
        W = weight_variable([ kernel_size, kernel_size, input_features.get_shape().as_list()[-1], outing_nb_features ])
        conv = tf.nn.conv2d(input_features, W, [ 1, 1, 1, 1 ], padding='SAME')
        if with_bias:
            conv= conv + bias_variable([ outing_nb_features ])
        return conv

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

def composite_function(input_features, out_features, kernel_size, is_training, keep_prob, name):
    '''Motivated by [12], we define H()
    as a composite function of three consecutive operations:
    batch normalization (BN) [14], followed by a rectified linear
    unit (ReLU) [6] and a 3 * 3 convolution (Conv).
    '''
    with tf.variable_scope('Composite_'+str(name)):
        preprocessed_features = tf.nn.relu(tf.layers.batch_normalization(input_features, training=is_training))
        new_features = conv2d(preprocessed_features, out_features, kernel_size)
        new_features = tf.nn.dropout(new_features, keep_prob)
        return new_features

def transition_up(input_features, bloc_idx):

    with tf.variable_scope('TransitionUp_'+str(bloc_idx)):
        #upscale encoded features
        transition=conv2d_upscale(input_features, 3)
        print('Transition up, upscaling encoded features {inshape}'.format(inshape=input_features.get_shape().as_list()))
        return transition

def transition_down(input_features, is_training, keep_prob, block_idx):
    '''The transition layers
    used in our experiments consist of a batch normalization
    layer and an 1*1 convolutional layer followed by a 2*2 average
    pooling layer. Actually, a ReLU is used, keeping the BN-ReLU-Conv structure
    ===> TODO, check if pooling could be replaced by learned convolutions (help reconstruct semantic map?)
    ------> would be conv2d with kernel=3 and strite=2 ?
    '''
    with tf.variable_scope('TransitionDown_'+str(block_idx)):
        preprocessed_features = tf.nn.relu(tf.layers.batch_normalization(input_features, training=is_training))
        new_features = conv2d(preprocessed_features, preprocessed_features.get_shape().as_list()[-1], kernel_size=1)
        new_features = tf.nn.dropout(new_features, keep_prob)
        transition = tf.nn.max_pool(new_features, ksize=[ 1, 2, 2, 1 ], strides=[1, 2, 2, 1 ], padding='SAME')
        return transition

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

def avg_pool(input, s):
  return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

def model(  data,
            n_outputs,
            hparams,
            mode):

    from tensorflow.contrib.learn import ModeKeys
    is_training = mode == ModeKeys.TRAIN

    #user config
    dropout_rate=0.2
    keep_prob=1.0-float(is_training)*dropout_rate

    #basic architexture for testing purpose
    '''nb_layers_sequence_encoding=[2, 3]
    bottleneck_nb_layers=5
    growth_rate=10
    output_only_inputs_last_decoding_block=True
    '''
    #FC-DenseNet-103 architecture:
    nb_layers_sequence_encoding=[4, 5, 7, 10]
    bottleneck_nb_layers=12
    growth_rate=16
    #set True in order to avoid the n-1 decoding block to be connected to the final conv outputs (False by default)
    output_only_inputs_last_decoding_block=False

    number_of_encoding_blocks=len(nb_layers_sequence_encoding)
    n_layers_per_block=nb_layers_sequence_encoding+[bottleneck_nb_layers]+nb_layers_sequence_encoding[::-1]
    print('Using {blocks} for each encoding and decoding branch with the following number of layers: {layers_per_blocks}'.format(blocks=number_of_encoding_blocks,
                                                                                                                                layers_per_blocks=n_layers_per_block))
    total_nb_layers=np.sum(n_layers_per_block)+len(n_layers_per_block)+2
    print('Expected number of layers (including input, output and transition blocks)= '+str(total_nb_layers))
    print('RAW data shape='+str(data.get_shape().as_list()))

    skip_connection_list = []
    field_of_view=1
    with tf.variable_scope('Encoder'):
        with tf.variable_scope('Input'):
            input_block_features_count=48
            first_filters_size=3
            feature_maps = conv2d(data, input_block_features_count, first_filters_size)
            #feature_maps=tf.nn.max_pool(input_features, [ 1, 3, 3, 1 ], strides=[1, 2, 2, 1 ], padding='SAME')
            print('Input block output shape='+str(feature_maps.get_shape().as_list()))
            field_of_view+=(first_filters_size-1)
        for blockID in xrange(number_of_encoding_blocks):
            feature_maps_block = block(feature_maps, n_layers_per_block[blockID], growth_rate, is_training, keep_prob, blockID)
            '''#keep the middle block to feed the contours detector
            if blockID==int(number_of_encoding_blocks/2):
                bottleneck_contours_input=feature_maps_block
            '''
            #concatenate with the input features
            feature_maps_block=tf.concat([feature_maps_block, feature_maps], axis=3, name='encoding_block_layers_concat_'+str(blockID))
            # At the end of the dense block, the current stack is stored in the skip_connections list
            skip_connection_list.append(feature_maps_block)
            feature_maps=transition_down( feature_maps_block, is_training, keep_prob, blockID)
            print('** concat(Block+input)+transition down shape='+str(feature_maps.get_shape().as_list()))
            #update field of view
            field_of_view+=n_layers_per_block[blockID]*2
        #reverse skip layers list to apply it in the decding layers set
        skip_connection_list = skip_connection_list[::-1]

    '''#central bottleneck
    with tf.variable_scope('Bottleneck_regions'):
        print('Central bottleneck with {central_nb_layers} layers'.format(central_nb_layers=n_layers_per_block[number_of_encoding_blocks]))
        last_encoding_feature_maps = block(feature_maps, n_layers_per_block[number_of_encoding_blocks], growth_rate, is_training, keep_prob, number_of_encoding_blocks)
        field_of_view+=n_layers_per_block[number_of_encoding_blocks]*2
    '''
    #central bottleneck installed at the end of the first half of the encoder blocks
    with tf.variable_scope('Bottleneck_contours'):
        print('Central bottleneck with {central_nb_layers} layers'.format(central_nb_layers=n_layers_per_block[number_of_encoding_blocks]))
        last_encoding_feature_maps = block(feature_maps, n_layers_per_block[number_of_encoding_blocks], growth_rate, is_training, keep_prob, number_of_encoding_blocks)
        field_of_view+=n_layers_per_block[number_of_encoding_blocks]*2

    #image classification task branch
    with tf.variable_scope('Classifier'):
        features_average = avg_pool(last_encoding_feature_maps, last_encoding_feature_maps.get_shape().as_list()[1])
        final_dim = last_encoding_feature_maps.get_shape().as_list()[-1]
        features_average_flat = tf.reshape(features_average, [ -1, final_dim ])
        Wfc = weight_variable([ final_dim, n_outputs ])
        bfc = bias_variable([ n_outputs ])
        logits_classif= tf.matmul(features_average_flat, Wfc) + bfc

    with tf.variable_scope('Regions_Segmentation'):
        # We store now the output of the next dense block in a list. We will only upsample these new feature maps
        block_to_upsample = []
        decoding_feature_maps=last_encoding_feature_maps
        for blockID in xrange(number_of_encoding_blocks):
            nlayers=n_layers_per_block[number_of_encoding_blocks + blockID+1]
            print('creating a block with {nlayers} layers'.format(nlayers=nlayers))

            # Transition Up ( Upsampling )
            feature_maps_up = transition_up(decoding_feature_maps, bloc_idx= blockID)

            #concatenate with the skip connection)
            print('Transition up, concatenated upscaled encoded features {upshape} with skip layer {skipshape}'.format(upshape=feature_maps_up.get_shape().as_list(),
                                                                                                                skipshape=skip_connection_list[blockID].get_shape().as_list()))
            feature_maps_in=tf.concat([feature_maps_up, skip_connection_list[blockID]], axis=3, name='decoding_block_layers_concat_'+str(blockID))
            print('Transition up+skip layers output features {outshape}'.format(outshape=feature_maps_in.get_shape().as_list()))
            # apply dense block
            decoding_feature_maps = block(feature_maps_in, nlayers, growth_rate, is_training, keep_prob, blockID)

        with tf.variable_scope('segmentation_output'):
            if output_only_inputs_last_decoding_block is True:
                logits_semantic_regions = conv2d(decoding_feature_maps, n_outputs, 1)
            else:
                logits_semantic_regions = conv2d(tf.concat([decoding_feature_maps, feature_maps_in], axis=3), n_outputs, 1)

    with tf.variable_scope('Contours_Segmentation'):
        # We store now the output of the next dense block in a list. We will only upsample these new feature maps
        block_to_upsample = []
        decoding_feature_maps=last_encoding_feature_maps
        for blockID in xrange(number_of_encoding_blocks):
            nlayers=n_layers_per_block[number_of_encoding_blocks + blockID+1]
            print('creating a block with {nlayers} layers'.format(nlayers=nlayers))

            # Transition Up ( Upsampling )
            feature_maps_up = transition_up(decoding_feature_maps, bloc_idx= blockID)

            #concatenate with the skip connection)
            print('Transition up, concatenated upscaled encoded features {upshape} with skip layer {skipshape}'.format(upshape=feature_maps_up.get_shape().as_list(),
                                                                                                                skipshape=skip_connection_list[blockID].get_shape().as_list()))
            feature_maps_in=tf.concat([feature_maps_up, skip_connection_list[blockID]], axis=3, name='decoding_block_layers_concat_'+str(blockID))
            print('Transition up+skip layers output features {outshape}'.format(outshape=feature_maps_in.get_shape().as_list()))
            # apply dense block
            decoding_feature_maps = block(feature_maps_in, nlayers, growth_rate, is_training, keep_prob, blockID)

        with tf.variable_scope('contours_output'):
            if output_only_inputs_last_decoding_block is True:
                logits_semantic_contours = conv2d(decoding_feature_maps, n_outputs, 1)
            else:
                logits_semantic_contours = conv2d(tf.concat([decoding_feature_maps, feature_maps_in], axis=3), n_outputs, 1)

    print('logits_semantic shape='+str(logits_semantic_regions.get_shape().as_list()))
    print('Last_encoding_feature_maps out shape = '+str(last_encoding_feature_maps.get_shape().as_list()))

    print('Semantic segmentation pixel field of view='+str(field_of_view))

    return {'code':last_encoding_feature_maps,
            'logits_semantic_regions_map':logits_semantic_regions,
            'logits_semantic_contours_map':logits_semantic_contours}
