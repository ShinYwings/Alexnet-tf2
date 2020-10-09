import numpy as np
from numpy import linalg as LA
import tensorflow as tf

MU = 0
SIGMA = 0.1

# 전체 트레이닝 데이터셋의 intensity를 구하는 거라
# 독자적으로 실행해서 구한 값을 더해줘야함...
def image_aug(img = "img", evecs_mat= "evecs_mat", evals= "evals"):
    
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

    return  I2
def intensity_RGB(images= "images"):

    res = np.zeros(shape=(1,3))
    for image in images:

        # Reshape the matrix to a list of rgb values.
        arr= image.reshape((256*256),3)
        # concatenate the vectors for every image with the existing list.
        res = np.concatenate((res,arr),axis=0)

    res = np.delete(res, (0), axis=0)           # 0번째 쉘 지우기
    
    m = res.mean(axis = 0)
    res = res - m
    
    R = np.cov(res, rowvar=False)
    evals, evecs = LA.eigh(R)
    
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    evals = tf.sqrt(evals)
    print("============")
    print(evals)
    # select the first 3 eigenvectors (3 is desired dimension
    # of rescaled data array)

    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    # perturbing color in image[0]
    # re-scaling from 0-1
    result = image_aug(images, evecs, evals)
    
    # return intensity value
    return result