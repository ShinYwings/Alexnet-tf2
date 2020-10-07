import tensorflow as tf 
import numpy as np
import os
import sys
import mnist_model as model
import time
import data_augmentation as da
from datetime import datetime as dt
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import optimizer_alexnet
import cv2
import threading
import progressbar
import math
import loadMetaData as lmd
# import sklearn
# from image_plotting import plot_confusion_matrix
# from image_plotting import plot_to_image

# Hyper parameters
# TODO : argparse?
LEARNING_RATE = 0.02
NUM_EPOCHS = 90
NUM_CLASSES = 10    # IMAGENET 2012
MOMENTUM = 0.9 # SGD + MOMENTUM
BATCH_SIZE = 128

LRN_INFO = (5, 1e-4, 0.75, 2) # radius, alpha, beta, bias   # hands-on 에서는 r=2 a = 0.00002, b = 0.75, k =1 이라고 되어있음...
INPUT_IMAGE_SIZE = 28 #WIDTH, HEIGHT    # cropped by 256x256 images
WEIGHT_DECAY = 5e-4

ENCODING_STYLE = "utf-8"
AUTO = tf.data.experimental.AUTOTUNE

with tf.device("/CPU:0"):
    def testtt(q, images, labels):

        test_images = list()
        test_labels = list()
                
        for i in range(0,len(labels)):

            # label = tf.cast(labels[i], tf.int32)
            # TODO with cpu 멀티프로세싱 해주기
            cropped_intend_image = image_cropping(images[i], training=False)
            # print(cropped_intend_image[0])
            for j in cropped_intend_image:
                # print("j",j)
                test_images.append(j)
                test_labels.append(labels[i])

        q.append((test_images, test_labels))


def image_cropping(images, training= None):
    
    INPUT_IMAGE_SIZE = 28

    cropped_images = list()

    if training:
        # intend_image = da.intensity_RGB(image=image)
        intend_images = tf.cast(images, tf.float32)
        
        # print(intend_image)
        horizental_fliped_image = tf.image.flip_left_right(intend_images)

        ran_crop_image1 = tf.image.random_crop(intend_images,size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1])
        ran_crop_image2 = tf.image.random_crop(horizental_fliped_image,
                                    size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1])

        cropped_images.append(ran_crop_image1)
        cropped_images.append(ran_crop_image2)
    
    else:
        horizental_fliped_images = tf.image.flip_left_right(images)
        # for original image
        topleft = tf.cast(images, dtype=tf.float32)
        
        cropped_images.append(topleft)
        
        # for horizental_fliped_image
        horizental_fliped_image_topleft = tf.cast(horizental_fliped_images, dtype=tf.float32)
        
        cropped_images.append(horizental_fliped_image_topleft)
        
    cropped_images = tf.stack(cropped_images)

    return cropped_images

def _parse_function(images, labels):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    # feature_description = {
    #     'image': tf.io.FixedLenFeature([], tf.string),
    #     'label': tf.io.FixedLenFeature([], tf.int64),
    # }
    # example = tf.io.parse_single_example(example_proto, feature_description)
    
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    labels = tf.cast(labels, tf.float32)
    # images = tf.subtract(images, IMAGENET_MEAN)
    # images = train_image_cropping(images)

    return images, labels

if __name__ == "__main__":
    
    root_dir=os.getcwd()
    sys.path.append(root_dir)

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    learning_rate_fn = optimizer_alexnet.AlexNetLRSchedule(initial_learning_rate = LEARNING_RATE, name="performance_lr")
    _optimizer = optimizer_alexnet.AlexSGD(learning_rate=learning_rate_fn, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, name="alexnetOp")
    # _optimizer = tf.keras.optimizers.Adam()
    _model = model.myModel(LRN_INFO)
    # 모델의 손실과 성능을 측정할 지표, 에포크가 진행되는 동안 수집된 측정 지표를 바탕으로 결과 출력
    # train_loss = tf.keras.metrics.MeanSquaredError(name= 'train_loss', dtype=tf.float32)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name= 'train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    prev_test_accuracy = tf.Variable(-1., trainable = False)

    with tf.device('/GPU:1'):
        @tf.function
        def train_step(images, labels):

            with tf.GradientTape() as tape:

                predictions = _model(images, training = True)
                loss = loss_object(labels, predictions)

            gradients = tape.gradient(loss, _model.trainable_variables)
            #apply gradients 가 v1의 minimize를 대체함
            _optimizer.apply_gradients(zip(gradients, _model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)
            
        @tf.function
        def test_step(test_images, test_labels):
            test_predictions = _model(test_images, training =False)
            t_loss = loss_object(test_labels, test_predictions)

            test_loss(t_loss)
            test_accuracy(test_labels, test_predictions)
        
        @tf.function
        def performance_lr_scheduling():
            learning_rate_fn.cnt_up_num_of_statinary_loss()
    # train_tensorboard= tf.keras.callbacks.TensorBoard(log_dir=train_logdir)
    # test_tensorboard= tf.keras.callbacks.TensorBoard(log_dir=val_logdir)

    # _model.build((None, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3))
    # _model.compile(optimizer=_optimizer, loss=loss_object, metrics=['accuracy'])
    # _model.fit(train_ds, steps_per_epoch=train_buf_size//BATCH_SIZE, validation_data=test_ds,validation_steps=test_buf_size//BATCH_SIZE ,epochs=NUM_EPOCHS, workers= 8, use_multiprocessing=True, callbacks=[train_tensorboard])
    # _model.summary()
    # _model.evaluate(test_ds, verbose=2, callbacks=[test_tensorboard])

    print("시작")
    for epoch in range(NUM_EPOCHS):
        start = time.perf_counter()
        
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

        q = list()

        isFirst = True
        for step, (images, labels) in enumerate(ds_train):

            if isFirst:
                t = threading.Thread(target=testtt, args=(q, images, labels))
                
                t.start()
                t.join()
                isFirst = False
            # TODO push ds
            else:
                
                train_images, train_labels = q.pop()
                train_images = tf.stack(train_images)
                train_labels = tf.stack(train_labels)
                train_batch_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
                train_batch_ds = train_batch_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
                
                t = threading.Thread(target=testtt, args=(q, images, labels))
                
                t.start()
                
                for batch_size_images, batch_size_labels in train_batch_ds:
                    print("in train_step",step)
                    train_step(batch_size_images, batch_size_labels)
                t.join()
        # Last step
        train_images, train_labels = q.pop()
        train_images = tf.stack(train_images)
        train_labels = tf.stack(train_labels)
        train_batch_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_batch_ds = train_batch_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
        for batch_size_images, batch_size_labels in train_batch_ds:
                    
            train_step(batch_size_images, batch_size_labels)
        
        # train_images = list()
            # train_labels = list()

            # for i in range(0,len(labels)):
                
            #     label = tf.cast(labels[i], tf.int32)
                
            #     # TODO with cpu 멀티프로세싱 해주기
            #     cropped_intend_image = image_cropping(images[i], training=True)
            #     # print(cropped_intend_image[0])
            #     for j in cropped_intend_image:
                    
            #         train_images.append(j)
            #         train_labels.append(label)
            
            # train_images = tf.stack(train_images)
            # train_labels = tf.stack(train_labels)
            
            # train_batch_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            # train_batch_ds = train_batch_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
            

        q2 = list()
        isFirst = True
        for step, (images, labels) in enumerate(ds_test):
            
            if isFirst:
                t = threading.Thread(target=testtt, args=(q2, images, labels))
                t.start()
                t.join()
                isFirst = False
            # TODO push ds
            else:
                test_images, test_labels = q2.pop()
                test_images = tf.stack(test_images)
                test_labels = tf.stack(test_labels)
                test_batch_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
                test_batch_ds = test_batch_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
                t = threading.Thread(target=testtt, args=(q2, images, labels))
                t.start()
                for batch_test_images, batch_test_labels in test_batch_ds:
                    print("in test_step",step)
                    test_step(batch_test_images, batch_test_labels)
                t.join()
        # Last step
        test_images, test_labels = q2.pop()
        test_images = tf.stack(test_images)
        test_labels = tf.stack(test_labels)
        test_batch_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_batch_ds = test_batch_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
        for batch_test_images, batch_test_labels in test_batch_ds:
                    
            test_step(batch_test_images, batch_test_labels)

        print('Epoch: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'.format(epoch+1,train_loss.result(),
                            train_accuracy.result()*100, test_loss.result(),test_accuracy.result()*100))
        
        print("Spends time({}) in Epoch {}".format(epoch+1, time.perf_counter() - start))

        if prev_test_accuracy >= test_accuracy.result():
            performance_lr_scheduling()
        prev_test_accuracy = test_accuracy.result()