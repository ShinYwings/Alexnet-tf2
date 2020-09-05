import tensorflow as tf 
import numpy as np
from sklearn.datasets import load_sample_image as img
from matplotlib import pyplot as plt
import skimage
import sys
import os
import cv2

FILTER_SIZE = 3
FILTER_NUM = 96

ROOT_DIR = os.path.abspath("D:\KingsCollege\seq1")
sys.path.append(ROOT_DIR)
file_list= os.listdir(ROOT_DIR)

image_list = list()

height, width, channels = [256, 455, 3]

#add images
for i in file_list:

    image_path = os.path.join(ROOT_DIR, i)
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    image_list.append(image)

images = np.array(image_list, dtype=np.float32)

batch_size, height, width, channels = images.shape

print("DHWC:",batch_size,", ",height,",",width,",",channels)

filters = np.zeros(shape=(FILTER_SIZE,FILTER_SIZE,channels, FILTER_NUM), dtype=np.float32)
# # filters[:,3,:,0] =1
# # filters[3,:,:,1] =1

convolve = lambda i, k: tf.nn.conv2d(i,k,strides=[1, 1, 1, 1], padding="SAME")

conv = convolve(images, filters)

relu = tf.nn.relu(conv, name="relu")

""" lrn """
lrn = tf.nn.local_response_normalization(relu, depth_radius=3, alpha=1e-4, beta=0.75, bias=2, name="lrn")

""" max pooling"""
mp = tf.nn.max_pool(lrn, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name="mp")

# outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")


# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = mp[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = plt.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
plt.show()