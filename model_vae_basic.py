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

# =============================== Q(z|X) ======================================
def Q(X, input_data_dim, z_dim, h_dim):
    with tf.variable_scope('Encoder'):
        with tf.variable_scope('layer_1'):
            #encoder parameters and graph
            Q_W1 = weight_variable([input_data_dim, h_dim])
            Q_b1 = bias_variable(shape=[h_dim])

            h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)

        with tf.variable_scope('layer_gaussian_mean'):
            Q_W2_mu = weight_variable([h_dim, z_dim])
            Q_b2_mu = bias_variable(shape=[z_dim])
            z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu

        with tf.variable_scope('layer_gaussian_std'):
            Q_W2_sigma = weight_variable([h_dim, z_dim])
            Q_b2_sigma = bias_variable(shape=[z_dim])
            z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma

        return z_mu, z_logvar


def sample_z(mu, log_var):
    with tf.name_scope('Distribution_sampler'):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================
def P(z, input_data_dim, z_dim, h_dim):
    with tf.variable_scope('Decoder'):
        #decoder parameters and graph
        with tf.variable_scope('layer_1'):
            P_W1 = weight_variable([z_dim, h_dim])
            P_b1 = bias_variable(shape=[h_dim])
            h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)

        with tf.variable_scope('layer_2'):
            P_W2 = weight_variable([h_dim, input_data_dim])
            P_b2 = bias_variable(shape=[input_data_dim])

            logits = tf.matmul(h, P_W2) + P_b2
            prob = tf.nn.sigmoid(logits)
        return prob, logits

# =============================== GLOBAL MODEL ====================================

def model(data,
            n_outputs,
            hparams,
            mode):

    #get input data dim
    data_initial_shape=data.get_shape().as_list()
    data_initial_shape[0]=-1
    data_flatten=tf.layers.flatten(data)
    X_dim=data_flatten.get_shape().as_list()[-1]

    #encode input data
    z_mu, z_logvar = Q(data_flatten, input_data_dim=X_dim, z_dim=n_outputs, h_dim=2*n_outputs)
    print('z_mu, z_logvar='+str((z_mu, z_logvar)))
    #sample from a gaussian setup by the codes
    z_sample = sample_z(z_mu, z_logvar)
    #generate
    _, logits = P(z_sample, input_data_dim=X_dim, z_dim=n_outputs, h_dim=2*n_outputs)

    #reshape to the initial shape
    logits=tf.reshape(logits,data_initial_shape)

    return {'z_mu':z_mu, 'z_logvar':z_logvar, 'code':tf.stack([z_mu, z_logvar], axis=1), 'reconstructed_data':logits}
