import tensorflow as tf 
import numpy as np
import os
import sys
import cifar_model as model
import time
import deprecated_data_generator as dg
from datetime import datetime as dt
from matplotlib import pyplot as plt
import optimizer_alexnet
import cv2

# Hyper parameters
# TODO : argparse?
LEARNING_RATE = 0.01
NUM_EPOCHS = 90
NUM_CLASSES = 10    # CIFAR-10
MOMENTUM = 0.9 # SGD + MOMENTUM
BATCH_SIZE = 128
DATASET_DIR = r"D:\cifar-10-batches-py"
LRN_INFO = (5, 1e-4, 0.75, 2) # radius, alpha, beta, bias   # hands-on 에서는 r=2 a = 0.00002, b = 0.75, k =1 이라고 되어있음...
INPUT_IMAGE_SIZE = 32 #WIDTH, HEIGHT    # cropped by 256x256 images
WEIGHT_DECAY = 5e-4
# TODO optimizer (weight decay & lr/10 heuristic 방법) 추가

# Fixed
DROUPUT_PROP = 0.5
ENCODING_STYLE = "utf-8"
AUTO = tf.data.experimental.AUTOTUNE

def get_logdir(root_logdir):
    run_id = dt.now().strftime("run_%Y_%m_%d-%H_%M_%S")
    
    return os.path.join(root_logdir, run_id)

if __name__ == "__main__":

    root_dir=os.getcwd()
    dataset_dir=os.path.abspath(DATASET_DIR)
    sys.path.append(root_dir)
    sys.path.append(dataset_dir)

    """Path for tf.summary.FileWriter and to store model checkpoints"""
    filewriter_path = os.path.join(root_dir, "tensorboard")
    checkpoint_path = os.path.join(root_dir, "checkpoints")

    """Create parent path if it doesn't exist"""
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    if not os.path.isdir(filewriter_path):
        os.mkdir(filewriter_path)

    file_list = dg.unpickle(dataset_dir)

    (train_images, train_labels), (test_images, test_labels) = dg.load_CIFAR10_data(file_list) 

    train_ds = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images,test_labels))
    train_ds = train_ds.shuffle(buffer_size=50000).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    test_ds = test_ds.shuffle(buffer_size=10000).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)

    _model = model.mAlexNet(INPUT_IMAGE_SIZE, LRN_INFO, NUM_CLASSES)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # TODO 실험중
    # lr_func = tf.keras.optimizers.schedules.LearningRateSchedule(lr_scheduler())
    
    # _optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    _optimizer = optimizer_alexnet.AlexSGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # _optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=False)
    # 모델의 손실과 성능을 측정할 지표, 에포크가 진행되는 동안 수집된 측정 지표를 바탕으로 결과 출력
    train_loss = tf.keras.metrics.Mean(name= 'train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    root_logdir = os.path.join(filewriter_path, "logs\\fit\\")

    logdir = get_logdir(root_logdir)

    summary_writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)
    
    print('tensorboard --logdir={}'.format(logdir))

    # _model.compile(optimizer=_optimizer, loss= loss_object, metrics=train_loss)
    # _model.fit(train_ds, epochs=NUM_EPOCHS, steps_per_epoch=20,
    #             batch_size=BATCH_SIZE, validation_batch_size=BATCH_SIZE, validation_data=test_ds)
    # _model.summary()

    with tf.device('/gpu:1'):
        @tf.function
        def train_step(images, labels):

            with tf.GradientTape() as tape:
                
                tape.watch(_model.trainable_variables)

                predictions = _model.call(images, training=True)
                loss = loss_object(labels, predictions)

            gradients = tape.gradient(loss, _model.trainable_variables)
            _optimizer.apply_gradients(zip(gradients, _model.trainable_variables))
            
            train_loss(loss)
            # train_accuracy(labels, predictions)
            train_accuracy.update_state(labels, predictions)

        @tf.function
        def test_step(images, labels):
            predictions = _model.call(images, training =False)
            t_loss = loss_object(labels, predictions)
            test_loss(t_loss)
            # test_accuracy(labels, predictions)
            test_accuracy.update_state(labels, predictions)
    
    print("시작")
    for epoch in range(NUM_EPOCHS):
        start = time.perf_counter()
        for images, labels in train_ds:
            train_step(images, labels)
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch+1)
            tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch+1)
            
        for images, labels in test_ds:
            test_step(images, labels)
        with summary_writer.as_default():
            tf.summary.scalar('test_loss', test_loss.result(), step=epoch+1)
            tf.summary.scalar('test_accuracy', test_accuracy.result(), step=epoch+1)
            
        print('에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'.format(epoch+1,train_loss.result(),
                            train_accuracy.result()*100, test_loss.result(),test_accuracy.result()*100))
        print("Epoch {} 의 총 소요시간: {}".format(epoch+1, time.perf_counter() - start))

        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
        
    print("끝")