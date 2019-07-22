import tensorflow as tf
import numpy as np
import h5py
import os
import cv2
from PIL import Image
path = "/home/robotlearning/PycharmProjects/AnimeHeroine/train"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


# Data processing


def save_image_to_h5py(path, name):
    img_list = []
    label_list = []
    dir_counter = 0
    with h5py.File(name+'.h5', 'w') as f:

        for child_dir in os.listdir(path):
            child_path = os.path.join(path, child_dir)

            for dir_image in os.listdir(child_path):
                img = cv2.imread(os.path.join(child_path, dir_image))
                temp = 'image/{}/{}'.format(child_dir, dir_image)
                f[temp] = img
                img_list.append(temp)
                label_list.append(dir_counter)

            dir_counter += 1
        label_np = np.array(label_list)
        f['labels'] = label_np

        f.close()
    return img_list


def main(argv=None):
    img_dir = save_image_to_h5py(path, 'train_set')     # 'img_dir' contain the paths of the images in the training set
    print(img_dir)


if __name__ == "__main__":
    tf.compat.v1.app.run()







