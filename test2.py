import math
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import multiprocessing
import tensorflow as tf
def tt(a="a",c="c" ,b="b"):
    with tf.device("/gpu:1"):
        print("a ",a,"b ",b,"c ",c)
    
if __name__ == "__main__":
    d = list()    
    a = 1
    b = 2
    c = 3