import tensorflow as tf 
import matplotlib.pyplot as plt
import cv2
import IPython.display


for raw_record in parsed_dataset:
    
    image = raw_record['image'].numpy()
    
    img = tf.image.decode_jpeg(image, channels=3)
    # img = tf.cast(img, tf.float32) / 255.0

    plt.imshow(img)
plt.show()