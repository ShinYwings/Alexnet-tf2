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
import shutil

# 독립적으로 실행되는 모듈
# Reference: https://www.kaggle.com/ryanholbrook/tfrecords-basics

TRAIN_IMAGE_DIR = r"D:\ILSVRC2012\ILSVRC2012_tfrecord_train"
# TEST_IMAGE_DIR = r"D:\ILSVRC2012\ILSVRC2012_tfrecord_val"
TRAIN_TFREC_DIR = r"D:\ILSVRC2012\class100_tfrecord_train"
# TEST_TFREC_DIR = r"D:\ILSVRC2012\class100_tfrecord_val"

def mv_tfrecord(meta_data = "meta_data", splited_dir_list = "splited_dir_list", 
                        tfrecord_dir= "tfrecord_dir", train=None,  mdir = "mdir",
                          image_dir = "image_dir", mindex = "mindex"):

    for dir_path in splited_dir_list:
        # os.chdir(new_dir)
        index = mdir.index(dir_path)
        index = mindex[index]
        print("index", index)
        if 440 <= index and index < 540:
            new_dir = os.path.join(tfrecord_dir, dir_path)
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
            print(index)
            print(dir_path)
            
            in_dir_path = os.path.join(image_dir, dir_path)
            all_file_list = os.listdir(in_dir_path)
            for file_name in all_file_list:

                # print("file_name", file_name)
                
                file_path = os.path.join(in_dir_path, file_name)

                mv_file_path = os.path.join(new_dir, file_name)

                # print("file_path", file_path)
                # print("mv_file_path", mv_file_path)
                shutil.copyfile(file_path, mv_file_path)
            
if __name__ == "__main__":

    # dir, index, name
    metadata = lmd.load_ILSVRC2012_metadata()

    _dir, _index, _name = metadata

    # train
    if not os.path.isdir(TRAIN_TFREC_DIR):
        os.mkdir(TRAIN_TFREC_DIR)
    os.chdir(TRAIN_TFREC_DIR)

    train_dir_list = os.listdir(TRAIN_IMAGE_DIR)

    # mv_tfrecord(meta_data = metadata, splited_dir_list = train_dir_list, 
    #                     tfrecord_dir= TRAIN_TFREC_DIR, train=None,  mdir = _dir,
    #                       image_dir = TRAIN_IMAGE_DIR, mindex = _index)
    split_number = math.ceil(len(train_dir_list) / 4)
    train_splited_dir_list = [train_dir_list[x:x + split_number] for x in range(0, len(train_dir_list), split_number)]
    
    p1 = Process(target=mv_tfrecord,
                args=(metadata, train_splited_dir_list[0], 
                    TRAIN_TFREC_DIR, True, _dir,
                    TRAIN_IMAGE_DIR, _index))
    p1.start()
    p2 = Process(target=mv_tfrecord,
                args=(metadata, train_splited_dir_list[1], 
                    TRAIN_TFREC_DIR, True, _dir,
                    TRAIN_IMAGE_DIR, _index))
    p2.start()
    p3 = Process(target=mv_tfrecord,
                args=(metadata, train_splited_dir_list[2], 
                    TRAIN_TFREC_DIR, True, _dir,
                    TRAIN_IMAGE_DIR, _index))
    p3.start()
    p4 = Process(target=mv_tfrecord,
                args=(metadata, train_splited_dir_list[3], 
                    TRAIN_TFREC_DIR, True, _dir,
                    TRAIN_IMAGE_DIR, _index))
    p4.start()
    # p5 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[4], 
    #                 TRAIN_TFREC_DIR, True, _dir,
    #                 TRAIN_IMAGE_DIR, _index))
    # p5.start()
    # p6 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[5], 
    #                 TRAIN_TFREC_DIR, True, _dir,
    #                 TRAIN_IMAGE_DIR, _index))
    # p6.start()
    # p7 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[6], 
    #                 TRAIN_TFREC_DIR, True, _dir,
    #                 TRAIN_IMAGE_DIR, _index))
    # p7.start()
    # p8 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[7], 
    #                 TRAIN_TFREC_DIR, True, _dir,
    #                 TRAIN_IMAGE_DIR, _index))
    # p8.start()

#test
    # if not os.path.isdir(TEST_TFREC_DIR):
    #     os.mkdir(TEST_TFREC_DIR)
    # os.chdir(TEST_TFREC_DIR)

    # train_dir_list = os.listdir(TEST_IMAGE_DIR)

    # # mv_tfrecord(meta_data = metadata, splited_dir_list = train_dir_list, 
    # #                     tfrecord_dir= TRAIN_TFREC_DIR, train=None,  mdir = _dir,
    # #                       image_dir = TRAIN_IMAGE_DIR, mindex = _index)
    # split_number = math.ceil(len(train_dir_list) / 8)
    # train_splited_dir_list = [train_dir_list[x:x + split_number] for x in range(0, len(train_dir_list), split_number)]
    
    # p1 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[0], 
    #                 TEST_TFREC_DIR, True, _dir,
    #                 TEST_IMAGE_DIR, _index))
    # p1.start()
    # p2 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[1], 
    #                 TEST_TFREC_DIR, True, _dir,
    #                 TEST_IMAGE_DIR, _index))
    # p2.start()
    # p3 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[2], 
    #                 TEST_TFREC_DIR, True, _dir,
    #                 TEST_IMAGE_DIR, _index))
    # p3.start()
    # p4 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[3], 
    #                 TEST_TFREC_DIR, True, _dir,
    #                 TEST_IMAGE_DIR, _index))
    # p4.start()
    # p5 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[4], 
    #                 TEST_TFREC_DIR, True, _dir,
    #                 TEST_IMAGE_DIR, _index))
    # p5.start()
    # p6 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[5], 
    #                 TEST_TFREC_DIR, True, _dir,
    #                 TEST_IMAGE_DIR, _index))
    # p6.start()
    # p7 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[6], 
    #                 TEST_TFREC_DIR, True, _dir,
    #                 TEST_IMAGE_DIR, _index))
    # p7.start()
    # p8 = Process(target=mv_tfrecord,
    #             args=(metadata, train_splited_dir_list[7], 
    #                 TEST_TFREC_DIR, True, _dir,
    #                 TEST_IMAGE_DIR, _index))
    # p8.start()