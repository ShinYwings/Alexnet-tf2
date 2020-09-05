import tensorflow as tf 
import numpy as np
import os
import sys 
import model
import train
import test
from data_generator import unpickle
from data_generator import load_CIFAR10_data as load_data
from data_generator import load_CIFAR10_meta as load_meta
from datetime import datetime
from matplotlib import pyplot as plt
from tensorflow.python.framework.ops import convert_to_tensor
import tensorflow.data as tfds
from tensorflow import keras


# Hyper parameters
# TODO : argparse?
INPUT_IMAGE_SIZE = 227 #WIDTH, HEIGHT  #x를 224x224로 해보고 그다음에 227x227로 바꾸기
LEARNING_RATE = 0.04
NUM_EPOCHS = 10
NUM_CLASSES = 10    # CIFAR-10
MOMENTUM = 0.9 # SGD + MOMENTUM
BATCH_SIZE = 128
DATASET_DIR = r"D:\cifar-10-batches-py" 

# Fixed
DROUPUT_PROP = 0.5
ENCODING_STYLE = "utf-8"

# How often we want to write the tf.summary data to disk
DISPLAY_STEP = 20

def preprocess_images(image, label):
    
    global INPUT_IMAGE_SIZE

    """N(0,1)로 norm"""
    image = tf.image.per_image_standardization(image)
    
    """resize images from 32X32 to INPUT_IMAGE_SIZE x INPUT_IMAGE_SIZE (alexnet standard)"""
    image = tf.image.resize(image, (INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE))

    return image, label

def get_run_logdir():
    run_id = datetime.time.strftime("run_%Y_%m_%d-%H_%M_%S")
    
    return os.path.join(root_logdir, run_id)

if __name__ == "__main__":

    # global INPUT_IMAGE_SIZE
    # global LEARNING_RATE
    # global NUM_EPOCHS
    # global BATCH_SIZE
    # global DROUPUT_PROP
    # global DISPLAY_STEP
    # global DATASET_DIR
    # global ENCODING_STYLE

    fc_layers= ['fc8', 'fc7', 'fc6']

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

    cifar_datasets = unpickle(dataset_dir)

    batch_size, class_names = load_meta(cifar_datasets[0])  # dataset의 첫번째 파일이 metadata 

    """[metadata, 5 * training batches, test batch]"""
    (train_images, train_labels), (test_images, test_labels) = load_data(cifar_datasets)

    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

    """check images are all right""" 
    # plt.figure(figsize=(20,20))

    # for i, (image,label) in enumerate(train_ds.take(5)):
    #     ax = plt.subplot(5,5,i+1)
    #     plt.imshow(image)
    #     plt.title(class_names[label].decode(ENCODING_STYLE))
    #     plt.axis('off')
    # plt.show()

    """
    Input Pipeline
    
    experimental: API for input pipelines
    cardinality: size of a set
        > in DB, 중복도가 낮으면 카디널리티가 높다. 중복도가 높으면 카디널리티가 낮다.
    """
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    val_ds_size = tf.data.experimental.cardinality(val_ds).numpy()

    print("Training data size: ", train_ds_size)
    print("Test data size: ", test_ds_size)
    print("Val data size: ", val_ds_size)

    """
    [3 primary operations]
        1. Preprocessing the data within the dataset
        2. Shuffle the dataset
        3. Batch data within the dataset
    
    drop_ramainder: 주어진 dataset을 batch_size 나눠주고 
                    batch_size 만족 못하는 나머지들을 남길지 버릴지

    prefetch(1): 데이터셋은 항상 한 배치가 미리 준비되도록 최선을 다합니다.
                 훈련 알고리즘이 한 배치로 작업을 하는 동안 이 데이터셋이 동시에 다음 배치를 준비
                 합니다. (디스크에서 데이터를 읽고 전처리)
    """
    train_ds = train_ds.map(preprocess_images, num_parallel_calls=5).shuffle(buffer_size=train_ds_size).batch(batch_size=BATCH_SIZE, drop_remainder=True)
    test_ds = (test_ds.map(preprocess_images, num_parallel_calls=5)
                            .shuffle(buffer_size=test_ds_size)
                            .batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(1))
    val_ds = (val_ds.map(preprocess_images, num_parallel_calls=5)
                            .shuffle(buffer_size=val_ds_size)
                            .batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(1))
    """
    Tensorboard

    monitoring
        - training loss
        - training accurarcy
        - validation loss
        - validation accuracy

    get_run_logdir: return the location of the exact directory that is named
                    according to the current time the training phase starts
    """

    root_logdir = os.path.join(os.curdir, "logs\\fit\\")

    run_logdir = get_run_logdir()

    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    """
    Training and Results

    To train the network, we have to compile it.

    Compilation processes
        - Loss function
        - Optimization Algorithm
        - Learning Rate
    """
    
    _iterator = iter(train_ds)

    batch = _iterator.get_next()

    alexnet = model.AlexNet(x,dropout_prob=DROUPUT_PROP, num_classes=NUM_CLASSES, fc_layer=fc_layers)
    
    # nesterov: SGD momentum은 가중치의 변화량에다가 모멘텀을 곱한 다음에 가중치를 업데이트를 해주는 반면,
    #           nesterov는 
    sgd = tf.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=False)
    train(alexnet, optimizer= sgd, dataset_train=train_ds, dataset_val=val_ds, epochs=NUM_EPOCHS)
    model.compile(optimizer=optimizer, loss=loss_fn)
    model.fit(dataset)
    keras.Model.fit(dataset)

    summary_writer = tf.summary.create_file_writer('/tmp/summaries')
    with summary_writer.as_default():
        tf.summary.scalar('loss', 0.1, step=DISPLAY_STEP)