# TensorFlow version of NIPS2016 soundnet
import tensorflow as tf


def conv2d(prev_layer, in_ch, out_ch, k_h=1, k_w=1, d_h=1, d_w=1, p_h=0, p_w=0, pad='VALID', name_scope='conv'):
    with tf.variable_scope(name_scope) as scope:
        # h x w x input_channel x output_channel
        w_conv = tf.get_variable('weights', [k_h, k_w, in_ch, out_ch],
                                 initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv = tf.get_variable('biases', [out_ch],
                                 initializer=tf.constant_initializer(0.0))

        padded_input = tf.pad(prev_layer, [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]], "CONSTANT") if pad == 'VALID' \
            else prev_layer

        output = tf.nn.conv2d(padded_input, w_conv,
                              [1, d_h, d_w, 1], padding=pad, name='z') + b_conv

        return output


def batch_norm(prev_layer, out_ch, eps, name_scope='conv'):
    with tf.variable_scope(name_scope) as scope:
        # mu_conv, var_conv = tf.nn.moments(prev_layer, [0, 1, 2], keep_dims=False)
        mu_conv = tf.get_variable('mean', [out_ch],
                                  initializer=tf.constant_initializer(0))
        var_conv = tf.get_variable('var', [out_ch],
                                   initializer=tf.constant_initializer(1))
        gamma_conv = tf.get_variable('gamma', [out_ch],
                                     initializer=tf.constant_initializer(1))
        beta_conv = tf.get_variable('beta', [out_ch],
                                    initializer=tf.constant_initializer(0))
        output = tf.nn.batch_normalization(prev_layer, mu_conv,
                                           var_conv, beta_conv, gamma_conv, eps, name='batch_norm')

        return output


def relu(prev_layer, name_scope='conv'):
    with tf.variable_scope(name_scope) as scope:
        return tf.nn.relu(prev_layer, name='a')


def maxpool(prev_layer, k_h=1, k_w=1, d_h=1, d_w=1, name_scope='conv'):
    with tf.variable_scope(name_scope) as scope:
        return tf.nn.max_pool(prev_layer,
                              [1, k_h, k_w, 1], [1, d_h, d_w, 1], padding='VALID', name='maxpool')


def dense(prev_layer, units, activation=tf.nn.leaky_relu, name_scope='retrain'):
    with tf.variable_scope(name_scope) as scope:
        return tf.layers.dense(prev_layer, units, activation=activation, kernel_initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))


def denseBN(prev_layer, name_scope):
    with tf.variable_scope(name_scope) as scope:
        return tf.layers.batch_normalization(prev_layer)
