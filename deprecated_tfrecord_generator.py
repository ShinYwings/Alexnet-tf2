import os
import sys
import cv2
import image_label_loader as dg
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import time

"""
input image : bytes
input label : int? string?
"""

_TRAIN_IMAGE_NUM = 50000
_TEST_IMAGE_NUM = 10000
_IMAGE_SIZE = 32
_CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]
DATASET_DIR = r"D:\cifar-10-batches-py"
TRAIN_IMAGE_DIR = r"D:\cifar-10-size-256\train"
TEST_IMAGE_DIR = r"D:\cifar-10-size-256\test"
TRAIN_FILE_NAME= lambda num : 'cifar-10-train_{}.tfrecord'.format(num)
TEST_FILE_NAME = lambda num : 'cifar-10-test_{}.tfrecord'.format(num)

def _int64_feature(value):
  # if not isinstance(values, (tuple, list)):
  #   values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))

def serialize_ds(image, label):
  feature_description = {
    'image': _bytes_feature(image),
    'label': _int64_feature(label),
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature_description))
  return example_proto.SerializeToString()

"""Loads data from the cifar10 pickle files and writes files to a TFRecord.
Args:
  filename: The filename of the cifar10 pickle file.
  tfrecord_writer: The TFRecord writer to use for writing.
  offset: An offset into the absolute number of images previously written.
Returns:
  The new offset.
"""

# 독립적으로 실행되는 모듈
# Reference: https://www.kaggle.com/ryanholbrook/tfrecords-basics

file_list= dg.unpickle(os.path.abspath(r'D:\cifar-10-batches-py'))

train_labels, test_labels = dg.load_ds(file_list)
train_images=dg.load_images(TRAIN_IMAGE_DIR)
test_images=dg.load_images(TEST_IMAGE_DIR)

print("finish loading all images")


os.chdir(r"D:\cifar-10-size-256\train_tfrecord")
for i in range(0,50000):
  with tf.io.TFRecordWriter(TRAIN_FILE_NAME(i)) as train_writer:
      print("parsing train",i,"번째 이미지")
      example = serialize_ds(train_images[i], train_labels[i])
      train_writer.write(example)

os.chdir(r"D:\cifar-10-size-256\test_tfrecord")
for j in range(0, 10000):
  with tf.io.TFRecordWriter(TEST_FILE_NAME(j)) as test_writer:
    print("parsing test",j,"번째 이미지")
    example = serialize_ds(test_images[j], test_labels[j])
    test_writer.write(example)
train_writer.close()
test_writer.close()
# with tf.python_io.TFRecordWriter(TRAIN_FILE_NAME) as tfrecord_writer:
#   offset = 0
#   for i in range(_NUM_TRAIN_FILES):
#     filename = os.path.join(dataset_dir,
#                             'cifar-10-batches-py',
#                             'data_batch_%d' % (i + 1))  # 1-indexed.
#     offset = _add_to_tfrecord(filename, tfrecord_writer, offset)

#   # Next, process the testing data:
#   with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
#     filename = os.path.join(dataset_dir,
#                             'cifar-10-batches-py',
#                             'test_batch')
#     _add_to_tfrecord(filename, tfrecord_writer)

  # Finally, write the labels file:
print('\nFinished converting the Cifar10 dataset!')

# train_images = list()
# train_labels = list()

# os.chdir(r"D:\cifar-10-size-256")

# tr_dir = os.path.join(r"D:\cifar-10-size-256", "train")
# ts_dir = os.path.join(r"D:\cifar-10-size-256", "test")

# os.chdir(tr_dir)
# os.chdir(ts_dir)