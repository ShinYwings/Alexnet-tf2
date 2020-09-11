import tensorflow as tf 
import numpy as np 
import os
import matplotlib.pyplot as plt
import cv2

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

    """
    Loaded in this way, each of the batch files contains a dictionary with the following elements:
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

    The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
    label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
    Binary version
    The binary version contains the files data_batch_1.bin, data_batch_2.bin, ..., data_batch_5.bin, as well as test_batch.bin. Each of these files is formatted as follows:
    <1 x label><3072 x pixel>
    ...
    <1 x label><3072 x pixel>
    In other words, the first byte is the label of the first image, which is a number in the range 0-9. The next 3072 bytes are the values of the pixels of the image. The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue. The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.

    Each file contains 10000 such 3073-byte "rows" of images, although there is nothing delimiting the rows. Therefore each file should be exactly 30730000 bytes long.

    There is another file, called batches.meta.txt. This is an ASCII file that maps numeric labels in the range 0-9 to meaningful class names. It is merely a list of the 10 class names, one per row. The class name on row i corresponds to numeric label i.
    """

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