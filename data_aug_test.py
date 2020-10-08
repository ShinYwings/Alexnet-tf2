import numpy as np
from numpy import linalg as LA
import tensorflow as tf 
import cv2
import pickle
import os

##################################
# test in colab environment!!!
##################################

ORIGINAL_IMAGE_SIZE = 256

def image_aug(img = "img", evecs_mat= "evecs_mat", evals= "evals"):
    
    img = tf.cast(img, tf.float32)

    mu = 0
    sigma = 0.1
    evecs_mat = evecs_mat
    evals = evals
    feature_vec=np.matrix(evecs_mat)

    # 3 x 1 scaled eigenvalue matrix
    # eval : eigenvalue
    # evec : eigenvector
    se = np.zeros(3)
    a1= np.random.normal(mu, sigma)    # random variable은 main함수에다 해줘야함X 
    a2 = np.random.normal(mu, sigma)   # 한 이미지가 트레이닝 한번 할때만 하는거니까
    a3= np.random.normal(mu, sigma)
    se[0] = a1* evals[0] 
    se[1] = a2* evals[1] 
    se[2] = a3* evals[2]
    se = np.matrix(se)
    I = tf.matmul(feature_vec, se.T)
    I = tf.cast(I, tf.float32)
    I2 = np.squeeze(I, axis=1)
    
    image = img + I2
    # print("I2 is",I2)
    # image = tf.cast(img, tf.float32)   # TODO float 32로 바꾸기
    # intensity = tf.fill((256,256,3), 128)   # TODO float 32로 바꾸기
    # Parse through every pixel value.
    # with tf.device("/gpu:1"):
    #     # for i in range(image.shape[0]):
    #     #     for j in range(image.shape[1]):
    #     #         # Parse through every dimension.
    #     #         for k in range(image.shape[2]):
    #                 # image[i,j,k] = image[i,j,k]+ I[k]
    #                 # intensity[i,j,k] = intensity[i,j,k]+ I[k]    
        
    #     image = image + I2
    #     intensity = intensity + I2
    # print()
    # print(image)
    # plt.imshow(intensity)
    # plt.show()
    return img, I2, image

path= r"/content/drive/My Drive/256_imgnet_val"
dirlist = os.listdir(path)
images = list()

for dir in dirlist[:10]:

  dir_path = os.path.join(path, dir)
  file_list = os.listdir(dir_path)

  for file_name in file_list:
      file_path = os.path.join(dir_path, file_name)
      with open(file_path, "rb") as f:
          raw_image = f.read()
          tf_image = tf.image.decode_jpeg(raw_image, channels=3)
          images.append(tf_image)

for image in images:

    res = np.zeros(shape=(1,3))
    # re-shape to make list of RGB vectors.
    arr=tf.reshape(image, shape=[(256*256),3])
    # consolidate RGB vectors of all images
    res = np.concatenate((res,arr),axis=0)
    res = np.delete(res, (0), axis=0)       # 0 인 쉘 지우기

    # Subtract the mean from each dimension
    # m = res.mean(axis = 0)
    # res = (res - m )
    # T = np.matmul(np.transpose(res), res)

    # 위에꺼 헷갈린듯? 왜냐면 np.cov에서 X -= avg[:, None] 있음
    R = np.cov(res, rowvar=False)

    # R : 256*256 X 3 mat
    # R^T * R  하면 3x3 mat 그다음 고유값 구하기  (대칭행렬)
    evals, evecs = LA.eigh(R)

    idx = np.argsort(evals)[::-1]  # 분산(=고유값)이 가장 큰 순서대로 인덱스화 하기
    evecs = evecs[:,idx]    # 분산이 가장 큰 순서대로 배열 (테스트는 거꾸로임)

    # sort eigenvectors according to same index

    evals = evals[idx]

    evals = tf.sqrt(evals)
    # select the first 3 eigenvectors (3 is desired dimension
    # of rescaled data array)

    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    # m = np.dot(_evecs.T, res.T).T

    # perturbing color in image[0]
    # re-scaling from 0-1

    img, I2, result = image_aug(image, evecs, evals)

    # print(result)
    if np.greater(np.max(result.numpy()), 255.0):
      print(np.max(img.numpy()),"  ",np.max(result.numpy()), "    ", I2)

print("끝")
    # print(result.numpy)