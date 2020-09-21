import tensorflow as tf 
import numpy as np 
import os
import matplotlib.pyplot as plt
import cv2

# data_generator에서 label만 가져오는 모듈

"""
    데이터 셋 importing 할때 256x256 으로 바로 resize 후 tf.data.dataset 적재하려고 했으나 실패 
"""

def unpickle(dir):   
        
    import pickle as pk

    file_list = list()

    all_file_list = os.listdir(dir)

    for i in all_file_list:

        file_path = os.path.join(dir, i)

        with open(file_path, 'rb') as f:
            file_list.append(pk.load(f, encoding='bytes'))

    return file_list

def load_images(dir):
    
    file_list = list()

    all_file_list = os.listdir(dir)

    for i in all_file_list:

        file_path = os.path.join(dir, i)
        with open(file_path, "rb") as image:
            f = image.read()
            file_list.append(f)
    
    return file_list

def load_ds(file_list):
        
    # batch_label = 5개 배치중 몇번째 배치인지
    # data : 한 배열마다 3072개 RGB 각 채널당 1024개  R->G->B 순서대로
    # labels: a list of 10000 numbers in the range 0-9. 
    #         The number at index i indicates the label of 
    #         the ith image in the array data.

    train_labels = list()
    test_labels = list()
    
    """
    training data generator
    """
    for j in range(1,6):

        _, labels, _, _ = file_list[j].values()
        
        for i in range(0,10000):
            
            loc = i+10000*(j-1)

            train_labels.append(labels[i])

    """
    test data generator
    """
    _, labels, _, _ = file_list[6].values()

    for i in range(0,10000):    
        
        test_labels.append(labels[i])
    
    return train_labels, test_labels