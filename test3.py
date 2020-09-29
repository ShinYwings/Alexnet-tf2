import datetime
import time
import multiprocessing
from multiprocessing import Queue
import tensorflow as tf 
import random
def worker(q, images, labels):
    # All cores are wrapped in cpu:0, i.e., TensorFlow does indeed use multiple CPU cores by default.
    with tf.device("/GPU:1"):
        for i in range(len(images)):
            # do something
            ans = images[i]+labels[i]
            time.sleep(10.)
            
    q.put(ans)

def main():

    test = 0.
    q = Queue()
    lock = lock()

    for j in range(10):
        images, labels = [i for i in range(127,-1,-1)], [i for i in range(0,128)]
        
        d = multiprocessing.Process(target = worker, args = (q, images, labels))
        d.daemon = True
        d.start()
        
    while(not q.isempty()):
        print(q.get())



if __name__ == "__main__":
    main()