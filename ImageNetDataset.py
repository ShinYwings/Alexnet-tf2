import os
import sys
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import time
import loadMetaData as lmd
import re
from threading import Thread as t
from multiprocessing import Process
import math

# 독립적으로 실행되는 모듈
# Reference: https://www.kaggle.com/ryanholbrook/tfrecords-basics

TRAIN_IMAGE_DIR = r"D:\ILSVRC2012\ILSVRC2012_img_train"
TEST_IMAGE_DIR = r"D:\ILSVRC2012\ILSVRC2012_img_val"
TRAIN_TFREC_DIR = r"D:\ILSVRC2012\ILSVRC2012_tfrecord_train"
TEST_TFREC_DIR = r"D:\ILSVRC2012\ILSVRC2012_tfrecord_val"
TRAIN_FILE_NAME= lambda name : '{}.tfrecord'.format(name)
TEST_FILE_NAME = lambda name : '{}.tfrecord'.format(name)
IMAGE_SIZE = 256
IMAGE_ENCODING_QUALITY = 70  # default 95
# TFRECORD_OPTION = tf.io.TFRecordOptions(compression_type="GZIP")

def _int64_feature(values):
  # if not isinstance(values, (tuple, list)):
  #   values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))

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

def convert_image_to_bytes(image="image"):
    #  이미지 받으면 RGB 순서로 받아짐. BGR로 받을지 안받을지는 알아서 결정해
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    height, width, _ = np.shape(image)  # return int
    
    # 한쪽만 256인 경우는 고려 안해줌. 256사이즈에 대해 그냥 resize하는 이유는 별로 시간 안걸리니까
    with tf.device('/gpu:1'):
      if width <= IMAGE_SIZE and height <= IMAGE_SIZE:
          cropped_img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
      
      elif width > height:
          center = (width - IMAGE_SIZE) / 2
          start = math.floor(center)
          end = math.ceil(center)
          resized_img = cv2.resize(image, (width, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
          cropped_img = resized_img[:,start:-end,:]
          
      else:
          center = (height - IMAGE_SIZE) / 2
          start = math.floor(center)
          end = math.ceil(center)
          resized_img = cv2.resize(image, (IMAGE_SIZE, height), interpolation=cv2.INTER_LINEAR)
          cropped_img = resized_img[start:-end,:,:]
      
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_ENCODING_QUALITY]
    is_success, im_buf_arr = cv2.imencode(".jpeg", cropped_img, encode_param)

    if is_success:
      img_raw = im_buf_arr.tobytes()
    else:
      print("image is not successfully converted")
      os.abort()

    return img_raw

def parse_to_tfrecord(meta_data = "meta_data", splited_dir_list = "splited_dir_list", 
                        tfrecord_dir= "tfrecord_dir", train=None,  mdir = "mdir",
                          image_dir = "image_dir", mindex = "mindex"):

    for dir_path in splited_dir_list:
        
        in_dir_path = os.path.join(image_dir, dir_path)

        all_file_list = os.listdir(in_dir_path)

        new_dir = os.path.join(tfrecord_dir, dir_path)

        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        os.chdir(new_dir)

        index = mdir.index(dir_path)
        # print(type(),"번째 index")
        # print("dir_path", dir_path)
        for file_name in all_file_list:
            
            file_path = os.path.join(in_dir_path, file_name)
            
            original_image = cv2.imread(file_path)
            image_bytes = convert_image_to_bytes(original_image)

            if train:
              with tf.io.TFRecordWriter(TRAIN_FILE_NAME(file_name.split(".")[0])) as writer:
                print("parsing train",file_name)
                example = serialize_ds(image_bytes, mindex[index])
                writer.write(example)

            else:
              with tf.io.TFRecordWriter(TEST_FILE_NAME(file_name.split(".")[0])) as writer:
                print("parsing test",file_name)
                example = serialize_ds(image_bytes, mindex[index])
                writer.write(example)

            writer.close()

    print("finish loading dataset of",image_dir)
    
if __name__ == "__main__":

  # dir, index, name
  metadata = lmd.load_ILSVRC2012_metadata()

  _dir, _index, _name = metadata

  #train
  if not os.path.isdir(TRAIN_TFREC_DIR):
      os.mkdir(TRAIN_TFREC_DIR)
  os.chdir(TRAIN_TFREC_DIR)

  train_dir_list = os.listdir(TRAIN_IMAGE_DIR)

  split_number = math.ceil(len(train_dir_list) / 8)
  train_splited_dir_list = [train_dir_list[x:x + split_number] for x in range(0, len(train_dir_list), split_number)]

  # TODO : put parse_to_tfrecord into p.starmap 
  p1 = Process(target=parse_to_tfrecord,
              args=(metadata, train_splited_dir_list[0], 
                TRAIN_TFREC_DIR, True, _dir,
                  TRAIN_IMAGE_DIR, _index))
  p2 = Process(target=parse_to_tfrecord,
              args=(metadata, train_splited_dir_list[1], 
                TRAIN_TFREC_DIR, True, _dir,
                  TRAIN_IMAGE_DIR, _index))
  p3 = Process(target=parse_to_tfrecord,
              args=(metadata, train_splited_dir_list[2], 
                TRAIN_TFREC_DIR, True, _dir,
                  TRAIN_IMAGE_DIR, _index))
  p4 = Process(target=parse_to_tfrecord,
              args=(metadata, train_splited_dir_list[3], 
                TRAIN_TFREC_DIR, True, _dir,
                  TRAIN_IMAGE_DIR, _index))
  p5 = Process(target=parse_to_tfrecord,
              args=(metadata, train_splited_dir_list[4], 
                TRAIN_TFREC_DIR, True, _dir,
                  TRAIN_IMAGE_DIR, _index))
  p6 = Process(target=parse_to_tfrecord,
              args=(metadata, train_splited_dir_list[5], 
                TRAIN_TFREC_DIR, True, _dir,
                  TRAIN_IMAGE_DIR, _index))
  p7 = Process(target=parse_to_tfrecord,
              args=(metadata, train_splited_dir_list[6], 
                TRAIN_TFREC_DIR, True, _dir,
                  TRAIN_IMAGE_DIR, _index))
  p8 = Process(target=parse_to_tfrecord,
              args=(metadata, train_splited_dir_list[7], 
                TRAIN_TFREC_DIR, True, _dir,
                  TRAIN_IMAGE_DIR, _index))

  #test
  # if not os.path.isdir(TEST_TFREC_DIR):
  #     os.mkdir(TEST_TFREC_DIR)
  # os.chdir(TEST_TFREC_DIR)

  # val_dir_list = os.listdir(TEST_IMAGE_DIR)
  
  # split_number = math.ceil(len(val_dir_list) / 8)
  # test_splited_dir_list = [val_dir_list[x:x + split_number] for x in range(0, len(val_dir_list), split_number)]

  # p1 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[0], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))
  # p2 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[1], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))
  # p3 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[2], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))
  # p4 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[3], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))
  # p5 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[4], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))        
  # p6 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[5], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))
  # p7 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[6], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))
  # p8 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[7], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))                                
  p1.start()
  p2.start()
  p3.start()
  p4.start()
  p5.start()
  p6.start()
  p7.start()
  p8.start()
  