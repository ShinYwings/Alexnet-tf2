import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform 
from numpy import linalg as LA
import tensorflow as tf 
import cv2

ORIGINAL_IMAGE_SIZE = 256

def image_aug(img = "img", evecs_mat= "evecs_mat", evals= "evals"):
    img = img
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
    _I = tf.matmul(feature_vec, se.T)
    _I = tf.cast(_I, tf.float32)
    I2 = np.squeeze(_I, axis=1)

    img = img + I2
    
    return img

def intensity_RGB(image= "image"):

    res = np.zeros(shape=(1,3))
    # re-shape to make list of RGB vectors.
    arr=tf.reshape(image, shape=[(ORIGINAL_IMAGE_SIZE*ORIGINAL_IMAGE_SIZE),3])
    # consolidate RGB vectors of all images
    res = np.concatenate((res,arr),axis=0)
    res = np.delete(res, (0), axis=0)       # 0번째 쉘 지우기

    R = np.cov(res, rowvar=False)

    evals, evecs = LA.eigh(R)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index

    evals = evals[idx]
    # select the first 3 eigenvectors (3 is desired dimension
    # of rescaled data array)

    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    # perturbing color in image[0]
    # re-scaling from 0-1
    
    result = image_aug(image, evecs, evals)
    # plt.imshow(img) 
    return result