import numpy as np
from numpy import linalg as LA
import tensorflow as tf

MU = 0
SIGMA = 0.1

# 전체 트레이닝 데이터셋의 intensity를 구하는 거라
# 독자적으로 실행해서 구한 값을 더해줘야함...
def image_aug(evecs_mat= "evecs_mat", evals= "evals"):
    
    feature_vec=np.matrix(evecs_mat)
    # 3 x 1 scaled eigenvalue matrix
    # eval : eigenvalue
    # evec : eigenvector
    se = np.zeros((3))
    a1= np.random.normal(MU, SIGMA)    # random variable은 main함수에다 해줘야함X 
    a2 = np.random.normal(MU, SIGMA)   # 한 이미지가 트레이닝 한번 할때만 하는거니까
    a3= np.random.normal(MU, SIGMA)
    se[0] = a1* evals[0]
    se[1] = a2* evals[1]
    se[2] = a3* evals[2]
    se = np.matrix(se)
    _I = tf.matmul(feature_vec, se.T)
    _I = tf.cast(_I, tf.float32)
    I2 = np.squeeze(_I, axis=1)
    print(I2)
    return  I2

def intensity_RGB(res= "res"):
    
    R = np.cov(res, rowvar=False)
    evals, evecs = LA.eigh(R)
    
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    evecs = evecs[:,:3]
    evals = tf.sqrt(evals)
    print("============")
    print(evals)
    # select the first 3 eigenvectors (3 is desired dimension
    # of rescaled data array)

    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    # perturbing color in image[0]
    # re-scaling from 0-1
    result = image_aug(evecs, evals)
    
    # return intensity value
    return result

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    raw_image= example['image']
    label= example['label']

    image = tf.image.decode_jpeg(raw_image, channels=3)
    image = tf.cast(image, tf.float32)
    #440은 imgnet metadata 상에 나와있는 index number 임. index 0부터 시작하게 만들려고 뺌
    label = tf.cast(tf.subtract(label,440), tf.int32)
    return image, label

import os
import sys

RUN_TRAIN_DATASET =  r"D:\ILSVRC2012\class10_tfrecord_train"

train_tfrecord_list = list()
test_tfrecord_list = list()

train_dirs = os.listdir(RUN_TRAIN_DATASET)

for train_dir in train_dirs:
    dir_path = os.path.join(RUN_TRAIN_DATASET, train_dir)
    a =tf.data.Dataset.list_files(os.path.join(dir_path, '*.tfrecord'))
    train_tfrecord_list.extend(a)

train_buf_size = len(train_tfrecord_list)

print("train_buf_size", train_buf_size)

train_ds = tf.data.TFRecordDataset(filenames=train_tfrecord_list, num_parallel_reads=tf.data.experimental.AUTOTUNE, compression_type="GZIP")
train_ds = train_ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

res = np.zeros(shape=(1,3))

for i,(image , _) in enumerate(train_ds):
    # Reshape the matrix to a list of rgb values.
    arr = tf.reshape(image,[(256*256),3]).numpy()
    # concatenate the vectors for every image with the existing list.
    res = np.concatenate((res,arr),axis=0)
    print(i)
res = np.delete(res, (0), axis=0)           # 0번째 쉘 지우기

m = res.mean(axis = 0)
res = res - m
result = intensity_RGB(res=res)

print(result)
