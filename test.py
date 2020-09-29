import tensorflow as tf 
import numpy as np
import os
import sys
import test2 as model
import time
import math
# data aug
import data_augmentation as da
from datetime import datetime as dt
from matplotlib import pyplot as plt
import optimizer_alexnet
import cv2
import multiprocessing
import progressbar

# Hyper parameters
# TODO : argparse?
LEARNING_RATE = 1e-3
NUM_EPOCHS = 90
NUM_CLASSES = 1000    # IMAGENET 2012
MOMENTUM = 0.9 # SGD + MOMENTUM
BATCH_SIZE = 128

DATASET_DIR = r"D:\ILSVRC2012"

TRAIN_TFRECORD_DIR = r"D:\ILSVRC2012\ILSVRC2012_tfrecord_train"
TEST_TFRECORD_DIR = r"D:\ILSVRC2012\ILSVRC2012_tfrecord_val"

# 학습 실험용
SAMPLE_TRAIN_TFRECORD_DIR = r"D:\ILSVRC2012\sample_tfrecord_train"
SAMPLE_TEST_TFRECORD_DIR = r"D:\ILSVRC2012\sample_tfrecord_val"

SAMPLE2_TRAIN_TFRECORD_DIR = r"D:\ILSVRC2012\20000_tfrecord_train"
SAMPLE2_TEST_TFRECORD_DIR = r"D:\ILSVRC2012\5000_tfrecord_val"

SAMPLE3_TRAIN_TFRECORD_DIR = r"D:\ILSVRC2012\20000_q95_tfrecord_train"
SAMPLE3_TEST_TFRECORD_DIR = r"D:\ILSVRC2012\5000_q95_tfrecord_val"

# 함수 실험용
FUNCTEST_TRAIN_TFRECORD_DIR = r"D:\ILSVRC2012\functest_tfrecord_train"
FUNCTEST_TEST_TFRECORD_DIR = r"D:\ILSVRC2012\functest_tfrecord_val"

# Input으로 넣을 데이터 선택
RUN_TRAIN_DATASET = SAMPLE2_TRAIN_TFRECORD_DIR
RUN_TEST_DATASET = SAMPLE2_TEST_TFRECORD_DIR

# 이건 핸즈온 책에서 참조 LRN_INFO = (2, 2e-5, 0.75, 1)
LRN_INFO = (5, 1e-4, 0.75, 2) # radius, alpha, beta, bias   # hands-on 에서는 r=2 a = 0.00002, b = 0.75, k =1 이라고 되어있음...
INPUT_IMAGE_SIZE = 227 #WIDTH, HEIGHT    # cropped by 256x256 images
WEIGHT_DECAY = 5e-4

# Fixed
IMAGENET_MEAN = [122.10927936917298, 116.5416959998387, 102.61744377213829] # rgb format
DROUPUT_PROP = 0.5
ENCODING_STYLE = "utf-8"
AUTO = tf.data.experimental.AUTOTUNE
CPU_CORE = multiprocessing.cpu_count()

widgets = [' [', 
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'), 
         '] ', 
           progressbar.Bar('/'),' (', 
           progressbar.ETA(), ') ', 

          ] 

def examine_image_cropping(image):
    global INPUT_IMAGE_SIZE
    
    ran_crop_image = tf.image.random_crop(image, 
                                            size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])
    _image = tf.cast(ran_crop_image, dtype=tf.float32)
    
    _image = tf.subtract(_image, IMAGENET_MEAN)

    return _image

def image_cropping(raw_image , training = None):  # do it only in test time
    
    global INPUT_IMAGE_SIZE

    cropped_images = list()

    horizental_fliped_image = tf.image.flip_left_right(image)

    if training:
        ran_crop_image1 = tf.image.random_crop(image,size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])
        ran_crop_image2 = tf.image.random_crop(horizental_fliped_image, 
                                    size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])

        # _image1 = tf.cast(ran_crop_image1, dtype=tf.float32)
        # _image2 = tf.cast(ran_crop_image2, dtype=tf.float32)

        cropped_images.append(tf.subtract(ran_crop_image1, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(ran_crop_image2, IMAGENET_MEAN))

    else:
        
        # for original image
        topleft = tf.cast(image[:227,:227], dtype=tf.float32)
        topright = tf.cast(image[29:,:227], dtype=tf.float32)
        bottomleft = tf.cast(image[:227,29:], dtype=tf.float32)
        bottomright = tf.cast(image[29:,29:], dtype=tf.float32)
        center = tf.cast(image[15:242, 15:242], dtype=tf.float32)

        cropped_images.append(tf.subtract(topleft, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(topright, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(bottomleft, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(bottomright, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(center, IMAGENET_MEAN))
        
        # for horizental_fliped_image
        horizental_fliped_image_topleft = tf.cast(horizental_fliped_image[:227,:227], dtype=tf.float32)
        horizental_fliped_image_topright = tf.cast(horizental_fliped_image[29:,:227], dtype=tf.float32)
        horizental_fliped_image_bottomleft = tf.cast(horizental_fliped_image[:227,29:], dtype=tf.float32)
        horizental_fliped_image_bottomright = tf.cast(horizental_fliped_image[29:,29:], dtype=tf.float32)
        horizental_fliped_image_center = tf.cast(horizental_fliped_image[15:242, 15:242], dtype=tf.float32)

        cropped_images.append(tf.subtract(horizental_fliped_image_topleft, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(horizental_fliped_image_topright, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(horizental_fliped_image_bottomleft, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(horizental_fliped_image_bottomright, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(horizental_fliped_image_center, IMAGENET_MEAN))
        
    return cropped_images

def get_logdir(root_logdir):
    run_id = dt.now().strftime("run_%Y_%m_%d-%H_%M_%S")
    
    return os.path.join(root_logdir, run_id)

def _parse_function(example_proto):

    print("exampple_proto", example_proto)
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    return example

# def _refine_function(example):

#     global BATCH_SIZE

#     _images = example['image']
#     _labels = example['label']
#     images = tf.stack(tf.zeros([1, 227,227,3]))
#     i = tf.constant(0)
#     tf.while_loop(lambda images, i: tf.less(i,128), 
#                    lambda  images, i: [tf.stack([images, examine_image_cropping(_images[i])]), tf.add(i, 1)],
#                     loop_vars=[images, i] , shape_invariants=[tf.TensorShape([None, 227, 227, 3]), i.get_shape()])
#     # tf.squeeze(images, axis=0)
#     tf.print(tf.shape(images))
#     labels = tf.cast(_labels-1, tf.int32)

#     return images, labels

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

    train_dirs = os.listdir(RUN_TRAIN_DATASET)
    test_dirs = os.listdir(RUN_TEST_DATASET)
    
    for train_dir in train_dirs:
        dir_path = os.path.join(RUN_TRAIN_DATASET, train_dir)
        a =tf.data.Dataset.list_files(os.path.join(dir_path, '*.tfrecord'))
        train_tfrecord_list.extend(a)
    
    for test_dir in test_dirs:
        dir_path = os.path.join(RUN_TEST_DATASET, test_dir)
        b = tf.data.Dataset.list_files(os.path.join(dir_path, '*.tfrecord'))
        test_tfrecord_list.extend(b)

    train_buf_size = len(train_tfrecord_list)
    test_buf_size= len(test_tfrecord_list)
    print("train_buf_size", train_buf_size)
    print("test_buf_size", test_buf_size)
    train_ds = tf.data.TFRecordDataset(filenames=train_tfrecord_list, num_parallel_reads=AUTO, compression_type="GZIP")
    test_ds = tf.data.TFRecordDataset(filenames=test_tfrecord_list, num_parallel_reads=AUTO, compression_type="GZIP")
    
    # train_ds = train_ds.shuffle(buffer_size=train_buf_size)
    # test_ds = test_ds.shuffle(buffer_size=test_buf_size)
    
    train_ds = train_ds.map(_parse_function, num_parallel_calls=AUTO)
    test_ds = test_ds.map(_parse_function, num_parallel_calls=AUTO)

    train_ds = train_ds.batch(batch_size=BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    test_ds = test_ds.batch(batch_size=BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    

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

    learning_rate_fn = optimizer_alexnet.AlexNetLRSchedule(initial_learning_rate = LEARNING_RATE)
    # _optimizer = optimizer_alexnet.AlexSGD(learning_rate=learning_rate_fn, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # _optimizer = optimizer_alexnet.AlexSGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    _optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    # _optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
    # 모델의 손실과 성능을 측정할 지표, 에포크가 진행되는 동안 수집된 측정 지표를 바탕으로 결과 출력
    train_loss = tf.keras.metrics.Mean(name= 'train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    prev_test_accuracy = -1.
    # NaN 발생이유 LR이 너무 높거나, 나쁜 초기화...
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
    train_logdir = os.path.join(logdir, "train\\")
    val_logdir = os.path.join(logdir, "val\\")

    """
    Training and Results

    To train the network, we have to compile it.

    Compilation processes
        - Loss function
        - Optimization Algorithm
        - Learning Rate
    """

    train_summary_writer = tf.summary.create_file_writer(train_logdir)
    val_summary_writer = tf.summary.create_file_writer(val_logdir)
    # tf.summary.trace_on(graph=True, profiler=True)
    
    print('tensorboard --logdir={}'.format(logdir))

    # _model.compile(optimizer=_optimizer, loss= loss_object, metrics=train_loss)
    # _model.fit(m_train_ds, epochs=NUM_EPOCHS, steps_per_epoch=20,
    #             batch_size=BATCH_SIZE, validation_batch_size=BATCH_SIZE, validation_data=m_test_ds)
    # _model.summary()

    prev_test_accuracy = tf.Variable(-1., trainable = False)
    
    with tf.device('/GPU:1'):

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = _model(images, training = True)

                loss = loss_object(labels, predictions)
            
            lr_var = _optimizer._decayed_lr(tf.float32)
            decay_rate = learning_rate_fn.decay_rate
            gradients = tape.gradient(loss, _model.trainable_variables)
            #apply gradients 가 v1의 minimize를 대체함
            _optimizer.apply_gradients(zip(gradients, _model.trainable_variables))
            
            train_loss(loss)
            train_accuracy(labels, predictions)

            tf.print(lr_var)
            # loss type : tensor
                
        # autograph 디버깅
        # print(tf.autograph.to_code(train_step.python_function))
        
        @tf.function
        def test_step(test_images, test_labels):
            test_predictions = _model(test_images, training = False)
            t_loss = loss_object(test_labels, test_predictions)
            test_loss(t_loss)
            test_accuracy(test_labels, test_predictions)
        @tf.function
        def performance_lr_scheduling():
            # tf.cond(tf.less_equal(test_accuracy.result(),prev_test_accuracy.read_value()),
            #     learning_rate_fn.cnt_up_num_of_statinary_loss,
            #     lambda: None)
            # prev_test_accuracy.assign(test_accuracy.result())
            learning_rate_fn.cnt_up_num_of_statinary_loss()
            
    print("시작")
    for epoch in range(NUM_EPOCHS):
        start = time.perf_counter()
        bar = progressbar.ProgressBar(max_value= math.ceil(train_buf_size/128.),  
                              widgets=widgets)
        test_bar = progressbar.ProgressBar(max_value= math.ceil(test_buf_size/128.),  
        widgets=widgets)
        bar.start()
        test_bar.start()

        for step, tb in enumerate(train_ds):
            
            raw_images= tb['image'].numpy()
            raw_labels= tb['label'].numpy()
            
            images = list()
            labels = list()
            print("len", len(raw_labels))

            for i in range(0,len(raw_labels)):

                image = tf.image.decode_jpeg(raw_images[i], channels=3)
                f_image = tf.cast(image, tf.float32)
                label = tf.cast(raw_labels[i]-1, tf.int32)
                intend_image = da.intensity_RGB(image=f_image)   # training 때만 적용

                # TODO with cpu 멀티프로세싱 해주기
                cropped_intend_image = image_cropping(intend_image, training=True)
                # cropped_intend_image = examine_image_cropping(image)

                for j in cropped_intend_image:
                    
                    images.append(j)
                    labels.append(label)
                
                # images.append(cropped_intend_image)
                # labels.append(label)
            
            images = tf.stack(images)
            labels = tf.stack(labels)

            train_batch_ds = tf.data.Dataset.from_tensor_slices((images, labels)).cache()
            train_batch_ds = train_batch_ds.shuffle(buffer_size=len(labels)).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
            
            for batch_size_images, batch_size_labels in train_batch_ds:

                train_step(batch_size_images, batch_size_labels)

            # train_step(images, labels)
            bar.update(step)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch+1)
            tf.summary.scalar('accuracy', train_accuracy.result()*100, step=epoch+1)
            
        for step, tc in enumerate(test_ds):
            test_raw_images= tc['image'].numpy()
            test_raw_labels= tc['label'].numpy()
            
            test_images = list()
            test_labels = list()

            print("len", len(test_raw_labels))

            for i in range(0,len(test_raw_labels)):
                test_image = tf.image.decode_jpeg(test_raw_images[i], channels=3)
                test_label = tf.cast(test_raw_labels[i]-1, tf.int32)
                # cropped_image= p.starmap(image_cropping, [(image, False)])
                
                # TODO with cpu 멀티프로세싱 해주기
                test_cropped_image = image_cropping(test_image, training=False)
                # cropped_image = examine_image_cropping(image)
                
                for k in test_cropped_image:
                    test_images.append(k)
                    test_labels.append(test_label)
                # images.append(cropped_image)
                # labels.append(label)
            test_images = tf.stack(test_images)
            test_labels = tf.stack(test_labels)
            #####
            batch_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).cache()
            batch_ds = batch_ds.shuffle(buffer_size=len(test_labels)).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
            
            for batch_size_images, batch_size_labels in batch_ds:

                test_step(batch_size_images, batch_size_labels)
            ####
            test_bar.update(step)
            
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch+1)
            tf.summary.scalar('accuracy', test_accuracy.result()*100, step=epoch+1)
            
        print('Epoch: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'.format(epoch+1,train_loss.result(),
                            train_accuracy.result()*100, test_loss.result(),test_accuracy.result()*100))
        
        print("Spends time({}) in Epoch {}".format(epoch+1, time.perf_counter() - start))

        if prev_test_accuracy >= test_accuracy.result():
            performance_lr_scheduling()
        prev_test_accuracy = test_accuracy.result()

        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
        
    print("끝")