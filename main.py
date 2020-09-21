import tensorflow as tf 
import numpy as np
import os
import sys
import model as model
import time
import data_augmentation as da
from datetime import datetime as dt
from matplotlib import pyplot as plt
import optimizer_alexnet
import cv2
import multiprocessing

# Hyper parameters
# TODO : argparse?
LEARNING_RATE = 0.01
NUM_EPOCHS = 90
NUM_CLASSES = 1000    # IMAGENET 2012
MOMENTUM = 0.9 # SGD + MOMENTUM
BATCH_SIZE = 128

DATASET_DIR = r"D:\ILSVRC2012"

TRAIN_TFRECORD_DIR = r"D:\ILSVRC2012\ILSVRC2012_tfrecord_train"
TEST_TFRECORD_DIR = r"D:\ILSVRC2012\ILSVRC2012_tfrecord_val"

# 함수 실험용
SAMPLE_TFRECORD_DIR = r"D:\ILSVRC2012\sample_tfrecord_train"
SAMPLE_TFRECORD_DIR = r"D:\ILSVRC2012\sample_tfrecord_val"

LRN_INFO = (5, 1e-4, 0.75, 2) # radius, alpha, beta, bias   # hands-on 에서는 r=2 a = 0.00002, b = 0.75, k =1 이라고 되어있음...
INPUT_IMAGE_SIZE = 227 #WIDTH, HEIGHT    # cropped by 256x256 images
WEIGHT_DECAY = 5e-4
# TODO optimizer (weight decay & lr/10 heuristic 방법) 추가

# Fixed
IMAGENET_MEAN = [122.10927936917298, 116.5416959998387, 102.61744377213829] # rgb format
DROUPUT_PROP = 0.5
ENCODING_STYLE = "utf-8"
AUTO = tf.data.experimental.AUTOTUNE
CPU_CORE = multiprocessing.cpu_count()
def image_cropping(image , training = None):  # do it only in test time
    
    global INPUT_IMAGE_SIZE

    cropped_images = list()

    horizental_fliped_image = tf.image.flip_left_right(image)
    
    # if test_mode:
    #     img = tf.image.resize(image, size=(227,227), method=tf.image.ResizeMethod.BILINEAR)
    #     img2 = tf.image.resize(horizental_fliped_image, size=(227,227), method=tf.image.ResizeMethod.BILINEAR)
        
        # cropped_images.append(tf.image.convert_image_dtype(img, dtype=tf.float32) - IMAGENET_MEAN)
        # cropped_images.append(tf.image.convert_image_dtype(img2, dtype=tf.float32) - IMAGENET_MEAN)         
        # return cropped_images

    if training:
        ran_crop_image1 = tf.image.random_crop(image,size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])
        ran_crop_image2 = tf.image.random_crop(horizental_fliped_image, 
                                    size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])

        _image1 = tf.image.convert_image_dtype(ran_crop_image1, dtype=tf.float32) - IMAGENET_MEAN
        _image2 = tf.image.convert_image_dtype(ran_crop_image2, dtype=tf.float32) - IMAGENET_MEAN
        cropped_images.append(_image1)
        cropped_images.append(_image2)
    else:
        
        # for original image
        topleft = tf.image.convert_image_dtype(image[:227,:227], dtype=tf.float32) - IMAGENET_MEAN
        topright = tf.image.convert_image_dtype(image[29:256,:227], dtype=tf.float32) - IMAGENET_MEAN
        bottomleft = tf.image.convert_image_dtype(image[:227,29:256], dtype=tf.float32) - IMAGENET_MEAN
        bottomright = tf.image.convert_image_dtype(image[29:256,29:256], dtype=tf.float32) - IMAGENET_MEAN
        center = tf.image.convert_image_dtype(image[15:242, 15:242], dtype=tf.float32) - IMAGENET_MEAN

        cropped_images.append(topleft)
        cropped_images.append(topright)
        cropped_images.append(bottomleft)
        cropped_images.append(bottomright)
        cropped_images.append(center)
    
        # for horizental_fliped_image
        horizental_fliped_image_topleft = tf.image.convert_image_dtype(horizental_fliped_image[:227,:227], dtype=tf.float32) - IMAGENET_MEAN
        horizental_fliped_image_topright = tf.image.convert_image_dtype(horizental_fliped_image[29:256,:227], dtype=tf.float32) - IMAGENET_MEAN
        horizental_fliped_image_bottomleft = tf.image.convert_image_dtype(horizental_fliped_image[:227,29:256], dtype=tf.float32) - IMAGENET_MEAN
        horizental_fliped_image_bottomright = tf.image.convert_image_dtype(horizental_fliped_image[29:256,29:256], dtype=tf.float32) - IMAGENET_MEAN
        horizental_fliped_image_center = tf.image.convert_image_dtype(horizental_fliped_image[15:242, 15:242], dtype=tf.float32) - IMAGENET_MEAN

        cropped_images.append(horizental_fliped_image_topleft)
        cropped_images.append(horizental_fliped_image_topright)
        cropped_images.append(horizental_fliped_image_bottomleft)
        cropped_images.append(horizental_fliped_image_bottomright)
        cropped_images.append(horizental_fliped_image_center)

    return cropped_images

def get_logdir(root_logdir):
    run_id = dt.now().strftime("run_%Y_%m_%d-%H_%M_%S")
    
    return os.path.join(root_logdir, run_id)

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    return example

# def _parse_function(example_proto):
    # # Parse the input `tf.train.Example` proto using the dictionary above.
    # feature_description = {
    #     'image': tf.io.FixedLenFeature([], tf.string),
    #     'label': tf.io.FixedLenFeature([], tf.int64),
    # }
    # example = tf.io.parse_single_example(example_proto, feature_description)
    # # a= np.array(, dtype=np.float32)
    
    # labels = example['label']
    
    # print("labels",labels)

    # raw_imgs= example['image']

    # a = list()
    # for j in range(len(raw_imgs)):
    #     raw_images = tf.io.decode_jpeg(raw_imgs[j], channels=3)
    #     a.append(raw_images)
    # # cropped_image = tf.image.resize(raw_images, (227, 227,3))
    # # raw_images = tf.image.decode_jpeg(example['image'].numpy(), channels=3)
    
    # b = list()
    # for i in a:
    #     print(i)
    #     cropped_image = tf.image.resize(i, [227,227], method= tf.image.ResizeMethod.BILINEAR)
    #     b.append(cropped_image)
    # print("images",b)
    # images = np.array(b, dtype=np.float32) / 255.0
    # print("images",images)
    
    # return (images, labels)

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
    
    train_tfrecord_list = list()
    test_tfrecord_list = list()

    train_dirs = os.listdir(TRAIN_TFRECORD_DIR)
    test_dirs = os.listdir(TEST_TFRECORD_DIR)
    
    for train_dir in train_dirs:
        dir_path = os.path.join(TRAIN_TFRECORD_DIR, train_dir)
        a =tf.data.Dataset.list_files(os.path.join(dir_path, '*.tfrecord'))
        train_tfrecord_list.extend(a)
    
    for test_dir in test_dirs:
        dir_path = os.path.join(TEST_TFRECORD_DIR, test_dir)
        b = tf.data.Dataset.list_files(os.path.join(dir_path, '*.tfrecord'))
        test_tfrecord_list.extend(b)

    train_buf_size = len(train_tfrecord_list)
    test_buf_size= len(test_tfrecord_list)
    
    train_ds = tf.data.TFRecordDataset(filenames=train_tfrecord_list, num_parallel_reads=AUTO, compression_type="GZIP")
    test_ds = tf.data.TFRecordDataset(test_tfrecord_list, num_parallel_reads=AUTO, compression_type="GZIP")
    train_ds = train_ds.map(_parse_function, num_parallel_calls=AUTO)
    test_ds = test_ds.map(_parse_function, num_parallel_calls=AUTO)
    train_ds = train_ds.shuffle(buffer_size=train_buf_size).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    test_ds = test_ds.shuffle(buffer_size=test_buf_size).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    
    """check images are all right""" 
    
    # plt.figure(figsize=(20,20))

    # for i, (image,_) in enumerate(train_ds.take(5)):
    #     ax = plt.subplot(5,5,i+1)
    #     plt.imshow(image[i])
    #     plt.axis('off')
    # plt.show()

    """
    Input Pipeline
    
    experimental: API for input pipelines
    cardinality: size of a set
        > in DB, 중복도가 낮으면 카디널리티가 높다. 중복도가 높으면 카디널리티가 낮다.
    """
    """
    [3 primary operations]
        1. Preprocessing the data within the dataset
        2. Shuffle the dataset
        3. Batch data within the dataset
    
    drop_ramainder: 주어진 dataset을 batch_size 나눠주고 
                    batch_size 만족 못하는 나머지들을 남길지 버릴지
    
    shuffle: Avoid local minima에 좋음
    
    prefetch(1): 데이터셋은 항상 한 배치가 미리 준비되도록 최선을 다합니다.
                 훈련 알고리즘이 한 배치로 작업을 하는 동안 이 데이터셋이 동시에 다음 배치를 준비
                 합니다. (디스크에서 데이터를 읽고 전처리)
    """


    _model = model.mAlexNet(INPUT_IMAGE_SIZE, LRN_INFO, NUM_CLASSES)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # TODO custom optimizer로 바꿔주기
    # _optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    _optimizer = optimizer_alexnet.AlexSGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # _optimizer = tf.keras.optimizers.AlexSGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # 모델의 손실과 성능을 측정할 지표, 에포크가 진행되는 동안 수집된 측정 지표를 바탕으로 결과 출력
    train_loss = tf.keras.metrics.Mean(name= 'train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    """
    Tensorboard

    monitoring
        - training loss
        - training accurarcy
        - validation loss
        - validation accuracy

    get_logdir: return the location of the exact directory that is named
                    according to the current time the training phase starts
    """

    root_logdir = os.path.join(filewriter_path, "logs\\fit\\")

    logdir = get_logdir(root_logdir)

    """
    Training and Results

    To train the network, we have to compile it.

    Compilation processes
        - Loss function
        - Optimization Algorithm
        - Learning Rate
    """

    summary_writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)
    
    print('tensorboard --logdir={}'.format(logdir))

    # _model.compile(optimizer=_optimizer, loss= loss_object, metrics=train_loss)
    # _model.fit(m_train_ds, epochs=NUM_EPOCHS, steps_per_epoch=20,
    #             batch_size=BATCH_SIZE, validation_batch_size=BATCH_SIZE, validation_data=m_test_ds)
    # _model.summary()

    with tf.device('/gpu:1'):
        @tf.function
        def train_step(train_model, images, labels):

            with tf.GradientTape() as tape:

                predictions = train_model.call(images, training=True)
                loss = loss_object(labels, predictions)

            gradients = tape.gradient(loss, train_model.trainable_variables)
            #apply gradients 가 v1의 minimize를 대체함
            _optimizer.apply_gradients(zip(gradients, train_model.trainable_variables))
            
            train_loss(loss)
            train_accuracy(labels, predictions)
            # train_accuracy.update_state(labels, predictions)

        @tf.function
        def test_step(test_model, images, labels):
            predictions = test_model.call(images, training =False)
            t_loss = loss_object(labels, predictions)
            test_loss(t_loss)
            test_accuracy(labels, predictions)
            # test_accuracy.update_state(labels, predictions)
    
    # p = multiprocessing.Pool(CPU_CORE)

    print("시작")
    for epoch in range(NUM_EPOCHS):
        start = time.perf_counter()
        for step, tb in enumerate(train_ds):
            
            raw_images= tb['image'].numpy()
            raw_labels= tb['label'].numpy()
            
            images = list()
            labels = list()

            for i in range(0,BATCH_SIZE):

                image = tf.image.decode_jpeg(raw_images[i], channels=3)

                # cropped_image= p.starmap(image_cropping, [(image, True)])
                
                cropped_image = image_cropping(image, training=True)
                for j in cropped_image:
                    images.append(j)
                    labels.append(tf.cast(raw_labels[i], tf.int32))

            images = tf.stack(images)
            labels = tf.stack(labels)
            
            train_step(_model, images, labels)
            # print("Training Epoch:", epoch+1, " Training Step:",step)
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch+1)
            tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch+1)
            tf.summary.scalar('learning_rate', _optimizer._decayed_lr(tf.float32), step=step)
            
        for step, tc in enumerate(test_ds):
            raw_images= tc['image']
            raw_labels= tc['label']
            
            images = list()
            labels = list()
            for i in range(0,BATCH_SIZE):
                image = tf.image.decode_jpeg(raw_images[i], channels=3)

                # intend_image = p.starmap(da.intensity_RGB, [(image)])
                # cropped_image= p.starmap(image_cropping, [(image, False)])
                # cropped_intend_image= p.starmap(image_cropping, [(intend_image, False)])

                intend_image = da.intensity_RGB(image=image)   # test때만 적용
                cropped_image = image_cropping(image, training=False)
                cropped_intend_image = image_cropping(intend_image, training=False)

                for j in cropped_image:
                    images.append(j)
                    labels.append(raw_labels[i])
                for j in cropped_intend_image:
                    images.append(j)
                    labels.append(raw_labels[i])
                
            images = tf.stack(images)
            labels = tf.stack(labels)
            
            test_step(_model, images, labels)
            # print("Testing Epoch:", epoch+1, " Testing Step:",step)
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