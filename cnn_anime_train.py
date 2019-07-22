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


# 1st conv2d layer
x = tf.placeholder(tf.float32, shape=(1, None, None, 3))
filter = tf.Variable(tf.random_normal([3, 3, 3, 7]))
op1_conv = tf.compat.v1.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME')

# 2nd conv2d layer
filter = tf.Variable(tf.random_normal([5, 5, 7, 21]))
op2_conv = tf.compat.v1.nn.conv2d(op1_conv, filter, strides=[1, 2, 2, 1], padding='SAME')

# 3rd max pooling layer
op3_max_pool = tf.compat.v1.nn.max_pool(op2_conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

# 4th conv2d layer
filter = tf.Variable(tf.random_normal([7, 1, 21, 128]))
op4_conv = tf.compat.v1.nn.conv2d(op3_max_pool, filter, strides=[1, 1, 1, 1], padding='SAME')

# 5th conv2d layer
filter = tf.Variable(tf.random_normal([1, 7, 128, 256]))
op5_conv = tf.compat.v1.nn.conv2d(op4_conv, filter, strides=[1, 1, 1, 1], padding='SAME')

# 6th max pooling layer
op6_max_pool = tf.compat.v1.nn.max_pool(op5_conv, [1, 4, 4, 1], [1, 4, 4, 1], padding='SAME')

# 7th con2d layer
filter = tf.Variable(tf.random_normal([3, 3, 256, 512]))
op7_conv = tf.compat.v1.nn.conv2d(op6_max_pool, filter, strides=[1, 1, 1, 1], padding='SAME')

# 8th con2d layer
filter = tf.Variable(tf.random_normal([5, 5, 512, 1024]))
op8_conv = tf.compat.v1.nn.conv2d(op7_conv, filter, strides=[1, 2, 2, 1], padding='SAME')

# 9th SPP layer
op9_SPP = SPP_layer(op8_conv, 3, pool_type='max')

# 10th fully connected layer




# Tensorboard
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()




# Session ----------------------------------------------------------------------------------------------------------


iterator = dataset_pair.make_initializable_iterator()
next_img = iterator.get_next()

print(sess.run(iterator.initializer))
while True:
    try:
        print(sess.run(next_img))
    except tf.errors.OutOfRangeError:
        break





