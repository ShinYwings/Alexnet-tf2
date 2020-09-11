import os
import cv2
import data_generator as dg
import tensorflow as tf
import Enlarge_Images as ei
import matplotlib.pyplot as plt
import numpy as np

DATASET_DIR = r"D:\cifar-10-batches-py"
dataset_dir=os.path.abspath(DATASET_DIR)

cifar_datasets = dg.unpickle(dataset_dir)

train_images = list()
train_labels = list()

os.chdir(r"D:\cifar-10-size-256")

tr_dir = os.path.join(r"D:\cifar-10-size-256", "train")
if not os.path.isdir(tr_dir):
    os.mkdir("train")
os.chdir(tr_dir)

for j in range(1,6):
    batch_label, labels, data, filenames = cifar_datasets[j].values()
    print("load ",batch_label.decode("utf-8"), "and adjust image size 32X32 to 256X256 (CIFAR-10)")
    # binary to img
    for i in range(0,10000):
        
        print("{}번째 이미지".format(i+10000*(j-1)))
        img = data[i].reshape((32,32,3), order="F")
        img = tf.image.per_image_standardization(img)
        brg_img = tf.reverse(img, axis=[-1])
        a = np.array(brg_img)
        y = cv2.resize(a, (256,256), interpolation=cv2.INTER_LINEAR)
        z = np.array(y, dtype=np.float32)
        cv2.normalize(z, z, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite("{}.jpg".format(i+10000*(j-1)), z)

# tr_dir = os.path.join(r"D:\cifar-10-size-256", "test")
# if not os.path.isdir(tr_dir):
#     os.mkdir("test")
# os.chdir(tr_dir)


# batch_label, labels, data, filenames = cifar_datasets[6].values()
# print("load ",batch_label.decode("utf-8"), "and adjust image size 32X32 to 256X256 (CIFAR-10)")
# # binary to img    
# for i in range(0,10000):
    
#     print("{}번째 이미지".format(i))
#     img = data[i].reshape((32,32,3), order="F")
#     img = tf.image.per_image_standardization(img)
#     brg_img = tf.reverse(img, axis=[-1])
#     a = np.array(brg_img)
#     y = cv2.resize(a, (256,256), interpolation=cv2.INTER_LINEAR)
#     z = np.array(y, dtype=np.float32)
#     cv2.normalize(z, z, 0, 255, cv2.NORM_MINMAX)
#     cv2.imwrite("{}.jpg".format(i), z)

# for i, image in enumerate(train_images):
#     ax = plt.subplot(10,10,i+1)
#     plt.imshow(image)
# plt.show()