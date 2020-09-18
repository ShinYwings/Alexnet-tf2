import pickle
import cv2
import os
import numpy as np

# 사실은 api에 있다고 한다.. ㅠ tf.image.resize

class resize_images():

    def __init__(self):
        self.dir_path= os.path.join(os.getcwd(),"resized_images")

        if not os.path.isdir(self.dir_path):
            os.mkdir(self.dir_path)
    # def batch(self, iterable, n=1):
    #     l = len(iterable)
    #     for ndx in range(0, l, n):
    #         yield iterable[ndx:min(ndx + n, l)]

    def call(self, images, train=bool):

        if not os.path.isdir(self.dir_path):

            if train:
                tr_dir = os.path.join(self.dir_path, "train")
                os.mkdir("train")
                os.chdir(tr_dir)
            else:
                ts_dir = os.path.join(self.dir_path, "test")
                os.mkdir("test")
                os.chdir(ts_dir)

        print("Ready for resizing images...")
        # for x in self.batch(images, 3):
        
        for i, x in enumerate(images):
            if train:
                name = 'resized_train_{}.jpg'.format(i)
            else:
                name = 'resized_test_{}.jpg'.format(i)
            
            a = np.array(x)
            y = cv2.resize(a, (256,256), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(name, y)
            print('created',name)
            
        # print("Finish conversion!")