import numpy as np
import tensorflow as tf


def SPP_layer(input, levels=3, name='SPP_layer', pool_type='max'):

    shape = input.get_shape().as_list()

    with tf.variable_scope(name):

        for l in range(levels):
            l = 2 ** l
            ksize = [1, np.ceil(shape[1] / l + 1).astype(np.int32), np.ceil(shape[2] / l + 1).astype(np.int32), 1]
            strides = [1, np.floor(shape[1] / l + 1).astype(np.int32), np.floor(shape[2] / l + 1).astype(np.int32), 1]

            if pool_type == 'max':
                pool = tf.compat.v1.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool, (shape[0], -1))

            else:
                pool = tf.compat.v1.nn.avg_pool(input, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool, (shape[0], -1))

            if l == 1:
                x_flatten = tf.reshape(pool, (shape[0], -1))

            else:
                x_flatten = tf.concat((x_flatten, pool), axis=1)

        print("SPP layer shape:\t", x_flatten.get_shape().as_list())


    return x_flatten


x = tf.ones((4,16,16,3))
x_sppl = SPP_layer(x,3)
