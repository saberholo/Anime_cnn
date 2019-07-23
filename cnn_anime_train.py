import tensorflow as tf
import numpy as np
import h5py
import os
import cv2
import joblib
from ETL import save_image_to_h5py
from ETL import path
from SPP import SPP_layer
name = 'train_set'
batch_size = 3


# Data Processing -----------------------------------------------------------------------------------------------------


img_dir = save_image_to_h5py(path, name)
f = h5py.File(name+'.h5', 'r')


source = []
for img in img_dir:
    source.append(f[img])
source = np.array(source)


def generator():
    for _ in source:
        yield _


sess = tf.Session()

dataset_img = tf.data.Dataset.from_generator(generator=generator, output_types=tf.int16)
# dataset_img = dataset_img.padded_batch(batch_size, padded_shapes=(None, None, None))
print(dataset_img)

dataset_label = tf.data.Dataset.from_tensor_slices(f['labels'])
# dataset_label = dataset_label.batch(batch_size)
print(dataset_label)

dataset_pair = tf.data.Dataset.zip((dataset_img, dataset_label))
print(dataset_pair)


# CNN Structure -----------------------------------------------------------------------------------------------------

x = tf.placeholder(tf.float32, shape=[1, None, None, 3])

# 1st conv2d layer
filter = tf.Variable(tf.random_normal([3, 3, 3, 7]))
op1_conv = tf.compat.v1.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME')
op1_relu = tf.compat.v1.nn.relu(op1_conv)

# 2nd conv2d layer
filter = tf.Variable(tf.random_normal([5, 5, 7, 21]))
op2_conv = tf.compat.v1.nn.conv2d(op1_relu, filter, strides=[1, 2, 2, 1], padding='SAME')
op2_relu = tf.compat.v1.nn.relu(op2_conv)

# 3rd max pooling layer
op3_max_pool = tf.compat.v1.nn.max_pool(op2_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

# 4th conv2d layer
filter = tf.Variable(tf.random_normal([7, 1, 21, 128]))
op4_conv = tf.compat.v1.nn.conv2d(op3_max_pool, filter, strides=[1, 1, 1, 1], padding='SAME')
op4_relu = tf.compat.v1.nn.relu(op4_conv)

# 5th conv2d layer
filter = tf.Variable(tf.random_normal([1, 7, 128, 256]))
op5_conv = tf.compat.v1.nn.conv2d(op4_relu, filter, strides=[1, 1, 1, 1], padding='SAME')
op5_relu = tf.compat.v1.nn.relu(op5_conv)

# 6th max pooling layer
op6_max_pool = tf.compat.v1.nn.max_pool(op5_relu, [1, 4, 4, 1], [1, 4, 4, 1], padding='SAME')

# 7th con2d layer
filter = tf.Variable(tf.random_normal([3, 3, 256, 512]))
op7_conv = tf.compat.v1.nn.conv2d(op6_max_pool, filter, strides=[1, 1, 1, 1], padding='SAME')
op7_relu = tf.compat.v1.nn.relu(op7_conv)

# 8th con2d layer
filter = tf.Variable(tf.random_normal([5, 5, 512, 1024]))
op8_conv = tf.compat.v1.nn.conv2d(op7_relu, filter, strides=[1, 2, 2, 1], padding='SAME')
op8_relu = tf.compat.v1.nn.relu(op8_conv)

# 9th SPP layer
# op9_SPP = SPP_layer(op8_relu, 3, pool_type='max')

# 10th fully connected layer
op9_SPP = tf.placeholder(tf.float32, shape=[None, ])


# Tensorboard
# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())
# writer.flush()


# Session ----------------------------------------------------------------------------------------------------------


sess.run(tf.global_variables_initializer())
iterator = dataset_pair.make_initializable_iterator()
next_img = iterator.get_next()

print(sess.run(iterator.initializer))
# while True:
#     try:
input = sess.run(next_img)
SPP_input = sess.run(op8_relu, feed_dict={x: [input[0]]})
op9_SPP_array = sess.run(SPP_layer(SPP_input, 3, pool_type='max'))    # SPP layer

print(sess.run(op9_SPP, feed_dict={op9_SPP: op9_SPP_array}))
    # except tf.errors.OutOfRangeError:
    #     break
    
    
    
    。。。。。
