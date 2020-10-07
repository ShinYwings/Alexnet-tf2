import tensorflow as tf 
import numpy as np
import os
import sys
import model3 as model
import time
import data_augmentation as da
from datetime import datetime as dt
from matplotlib import pyplot as plt
import optimizer_alexnet
import cv2
import multiprocessing
import progressbar
import math
import loadMetaData as lmd
# import sklearn
# from image_plotting import plot_confusion_matrix
# from image_plotting import plot_to_image

# Hyper parameters
# TODO : argparse?
LEARNING_RATE = 5e-4
NUM_EPOCHS = 90
NUM_CLASSES = 1000    # IMAGENET 2012
MOMENTUM = 0.9 # SGD + MOMENTUM
BATCH_SIZE = 64

DATASET_DIR = r"D:\ILSVRC2012"


# 본 게임
TRAIN_TFRECORD_DIR = r"D:\ILSVRC2012\ILSVRC2012_tfrecord_train"
TEST_TFRECORD_DIR = r"D:\ILSVRC2012\ILSVRC2012_tfrecord_val"

#image만
SAMPLE_TRAIN_IMAGES_DIR = r"D:\ILSVRC2012\256_imgnet_train"
SAMPLE_TEST_IMAGES_DIR = r"D:\ILSVRC2012\256_imgnet_val"


# 학습 실험용
SAMPLE_TRAIN_TFRECORD_DIR = r"D:\ILSVRC2012\sample_tfrecord_train"
SAMPLE_TEST_TFRECORD_DIR = r"D:\ILSVRC2012\sample_tfrecord_val"


SAMPLE3_TRAIN_TFRECORD_DIR = r"D:\ILSVRC2012\20000_q95_tfrecord_train"
SAMPLE3_TEST_TFRECORD_DIR = r"D:\ILSVRC2012\5000_q95_tfrecord_val"


# 함수 실험용
FUNCTEST_TRAIN_TFRECORD_DIR = r"D:\ILSVRC2012\functest_tfrecord_train"
FUNCTEST_TEST_TFRECORD_DIR = r"D:\ILSVRC2012\functest_tfrecord_val"

# Input으로 넣을 데이터 선택
RUN_TRAIN_DATASET = SAMPLE_TRAIN_IMAGES_DIR
RUN_TEST_DATASET = SAMPLE_TEST_IMAGES_DIR

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

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
# for x in batch(range(0, 10), 3):
#     print x
def train_image_cropping(images):
    
    INPUT_IMAGE_SIZE = 227

    cropped_images = list()

    #TODO intensity 바꾸기 아래는 테스트용
    # intend_image = da.intensity_RGB(image=image)
    intend_images = tf.cast(images, tf.float32)
    
    # print(intend_image)
    horizental_fliped_image = tf.image.flip_left_right(intend_images)

    ran_crop_image1 = tf.image.random_crop(intend_images,size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])
    ran_crop_image2 = tf.image.random_crop(horizental_fliped_image,
                                size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])

    test = tf.subtract(ran_crop_image1, IMAGENET_MEAN)/255.
    # print(test)
    cropped_images.append(test)
    cropped_images.append(tf.subtract(ran_crop_image2, IMAGENET_MEAN)/255.)
    
    cropped_images = tf.stack(cropped_images)

    return cropped_images

def test_image_cropping(images):
    
    INPUT_IMAGE_SIZE = 227

    cropped_images = list()

    horizental_fliped_images = tf.image.flip_left_right(images)
    # for original image
    topleft = tf.cast(images[:227,:227,:], dtype=tf.float32)
    topright = tf.cast(images[29:,:227,:], dtype=tf.float32)
    bottomleft = tf.cast(images[:227,29:,:], dtype=tf.float32)
    bottomright = tf.cast(images[29:,29:,:], dtype=tf.float32)
    center = tf.cast(images[15:242, 15:242,:], dtype=tf.float32)

    cropped_images.append(tf.subtract(topleft, IMAGENET_MEAN)/255.)
    cropped_images.append(tf.subtract(topright, IMAGENET_MEAN)/255.)
    cropped_images.append(tf.subtract(bottomleft, IMAGENET_MEAN)/255.)
    cropped_images.append(tf.subtract(bottomright, IMAGENET_MEAN)/255.)
    cropped_images.append(tf.subtract(center, IMAGENET_MEAN)/255.)
    
    # for horizental_fliped_image
    horizental_fliped_image_topleft = tf.cast(horizental_fliped_images[:227,:227,:], dtype=tf.float32)
    horizental_fliped_image_topright = tf.cast(horizental_fliped_images[29:,:227,:], dtype=tf.float32)
    horizental_fliped_image_bottomleft = tf.cast(horizental_fliped_images[:227,29:,:], dtype=tf.float32)
    horizental_fliped_image_bottomright = tf.cast(horizental_fliped_images[29:,29:,:], dtype=tf.float32)
    horizental_fliped_image_center = tf.cast(horizental_fliped_images[15:242, 15:242,:], dtype=tf.float32)

    cropped_images.append(tf.subtract(horizental_fliped_image_topleft, IMAGENET_MEAN)/255.)
    cropped_images.append(tf.subtract(horizental_fliped_image_topright, IMAGENET_MEAN)/255.)
    cropped_images.append(tf.subtract(horizental_fliped_image_bottomleft, IMAGENET_MEAN)/255.)
    cropped_images.append(tf.subtract(horizental_fliped_image_bottomright, IMAGENET_MEAN)/255.)
    cropped_images.append(tf.subtract(horizental_fliped_image_center, IMAGENET_MEAN)/255.)
    
    cropped_images = tf.stack(cropped_images)

    return cropped_images

def get_logdir(root_logdir):
    run_id = dt.now().strftime("run_%Y_%m_%d-%H_%M_%S")
    
    return os.path.join(root_logdir, run_id)

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
    
    root_logdir = os.path.join(filewriter_path, "logs\\fit\\")
    logdir = get_logdir(root_logdir)
    train_logdir = os.path.join(logdir, "train\\")
    val_logdir = os.path.join(logdir, "val\\")

    train_images_list = list()
    train_labels_list = list()
    test_images_list = list()
    test_labels_list = list()

    metadata = lmd.load_ILSVRC2012_metadata()

    _dir, _index, _name = metadata

    train_dirs = os.listdir(RUN_TRAIN_DATASET)
    test_dirs = os.listdir(RUN_TEST_DATASET)
    
    for train_dir in train_dirs:
        dir_path = os.path.join(RUN_TRAIN_DATASET, train_dir)
        label_name = train_dir
        index = _dir.index(train_dir)
        print(train_dir)
        print(_index[index])
        open_files = os.listdir(dir_path)
        for file_name in open_files:
            
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, "rb") as f:
                raw_image = f.read()
                image = tf.image.decode_jpeg(raw_image, channels=3)
                label = _index[index]-1
                train_images_list.append(image)
                train_labels_list.append(label)
    
    for test_dir in test_dirs:
        dir_path = os.path.join(RUN_TEST_DATASET, test_dir)
        label_name = test_dir
        index = _dir.index(test_dir)
        print(test_dir)
        print(_index[index])

        open_files = os.listdir(dir_path)
        for file_name in open_files:
            
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, "rb") as f:
                raw_image = f.read()
                image = tf.image.decode_jpeg(raw_image, channels=3)
                label = _index[index]-1
                test_images_list.append(image)
                test_labels_list.append(label)
                print("load",file_name,"in",test_dir,"index is",_index[index])
                
    train_buf_size = len(train_images_list)
    test_buf_size= len(test_images_list)
    print("train_buf_size", train_buf_size, "\t", len(train_labels_list))
    print("test_buf_size", test_buf_size, "\t", len(test_labels_list))

    # train_images_list = tf.stack(train_images_list)
    # train_labels_list = tf.stack(train_labels_list)
    # test_images_list = tf.stack(test_images_list)
    # test_labels_list = tf.stack(test_labels_list)

    # train_images_list = tf.cast(train_images_list, tf.float32)
    # train_labels_list = tf.cast(train_labels_list, tf.int32)
    # test_images_list = tf.cast(test_images_list, tf.float32)
    # test_labels_list = tf.cast(test_labels_list, tf.int32)
    # print(train_images_list[1])
    # print(tf.shape(train_images_list))
    # print(tf.shape(test_images_list))
    # print(np.shape(train_images_list))

    # train_ds = tf.data.Dataset.from_tensor_slices((train_images_list, train_labels_list))
    # test_ds = tf.data.Dataset.from_tensor_slices((test_images_list, test_labels_list))

    # train_ds = train_ds.shuffle(buffer_size=train_buf_size)
    # test_ds = test_ds.shuffle(buffer_size=test_buf_size)
    # train_ds = train_ds.map(_parse_function, num_parallel_calls=AUTO)
    # test_ds = test_ds.map(_parse_function, num_parallel_calls=AUTO)
    # train_ds = train_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    # test_ds = test_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)

    """check images are all right"""
    
    # plt.figure(figsize=(20,20))

    # for i, (image,_) in enumerate(train_ds.take(5)):
    #     ax = plt.subplot(5,5,i+1)
    #     plt.imshow(image[i])
    #     plt.axis('off')
    # plt.show()
    
    _model = model.mAlexNet(INPUT_IMAGE_SIZE, NUM_CLASSES)
    # _model.build((None,INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE,3))
    
    # learning_rate_fn = optimizer_alexnet.AlexNetLRSchedule(initial_learning_rate = LEARNING_RATE, name="performance_lr")
    # _optimizer = optimizer_alexnet.AlexSGD(learning_rate=learning_rate_fn, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, name="alexnetOp")
    _optimizer = tf.keras.optimizers.Adam()
    # 모델의 손실과 성능을 측정할 지표, 에포크가 진행되는 동안 수집된 측정 지표를 바탕으로 결과 출력
    # train_loss = tf.keras.metrics.MeanSquaredError(name= 'train_loss', dtype=tf.float32)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name= 'train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    train_summary_writer = tf.summary.create_file_writer(train_logdir)
    val_summary_writer = tf.summary.create_file_writer(val_logdir)
    
    # tf.summary.trace_on(graph=True)
    
    print('tensorboard --logdir={}'.format(logdir))

    prev_test_accuracy = tf.Variable(-1., trainable = False)

    train_tensorboard= tf.keras.callbacks.TensorBoard(log_dir=train_logdir)
    test_tensorboard= tf.keras.callbacks.TensorBoard(log_dir=val_logdir)

    # _model.build((None, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3))
    _model.compile(optimizer=_optimizer, loss=loss_object, metrics=['accuracy'])
    _model.fit(train_ds, steps_per_epoch=train_buf_size//BATCH_SIZE, validation_data=test_ds,validation_steps=test_buf_size//BATCH_SIZE ,epochs=NUM_EPOCHS, workers= 8, use_multiprocessing=True, callbacks=[train_tensorboard])
    _model.summary()
    # _model.evaluate(test_ds, verbose=2, callbacks=[test_tensorboard])
    # print("시작")
    # for epoch in range(NUM_EPOCHS):
    #     start = time.perf_counter()
    #     bar = progressbar.ProgressBar(max_value= math.ceil(train_buf_size/128.), widgets=widgets)
    #     test_bar = progressbar.ProgressBar(max_value= math.ceil(test_buf_size/128.),  
    #     widgets=widgets)
    #     bar.start()
        
    #     for step, (images, labels) in enumerate(train_ds):

    #         # train_images = list()
    #         # train_labels = list()
    #         # print(len(images))
    #         # print(len(labels))

    #         # for i in range(0,len(labels)):
                
    #         #     # print("raw", image)
    #         #     # label = tf.cast(raw_labels[i]-1, tf.int32)
                
    #         #     label = labels[i]
    #         #     # TODO with cpu 멀티프로세싱 해주기
    #         #     cropped_intend_image = image_cropping(images[i], training=True)
    #         #     # print(cropped_intend_image[0])
    #         #     for j in cropped_intend_image:
    #         #         # print("j",j)
    #         #         train_images.append(j)
    #         #         train_labels.append(label)
            
    #         # train_images = tf.stack(train_images)
    #         # train_labels = tf.stack(train_labels)
            
    #         # # print(images[4])
    #         # # print("images", images)
    #         # train_batch_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    #         # train_batch_ds = train_batch_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
            
    #         # for batch_size_images, batch_size_labels in train_batch_ds:
                
    #         #     # print(batch_size_images.shape, batch_size_labels.shape)

    #         #     train_step(batch_size_images, batch_size_labels)
    #         images = tf.cast(images, tf.float32)
    #         train_step(images, labels)
    #         bar.update(step)

    #     with train_summary_writer.as_default():
    #         tf.summary.scalar('loss', train_loss.result(), step=epoch+1)
    #         tf.summary.scalar('accuracy', train_accuracy.result()*100, step=epoch+1)
        
    #     test_bar.start()
    #     for step, (images, labels) in enumerate(test_ds):

    #         # test_images = list()
    #         # test_labels = list()
    #         # print(len(images))
    #         # print(len(labels))

    #         # for i in range(0,len(labels)):
                
    #         #     # print("raw", image)
    #         #     # label = tf.cast(raw_labels[i]-1, tf.int32)
                
    #         #     label = labels[i]
    #         #     # TODO with cpu 멀티프로세싱 해주기
    #         #     cropped_intend_image = image_cropping(images[i], training=False)
    #         #     # print(cropped_intend_image[0])
    #         #     for j in cropped_intend_image:
    #         #         # print("j",j)
    #         #         test_images.append(j)
    #         #         test_labels.append(label)

    #         # test_images = tf.stack(test_images)
    #         # test_labels = tf.stack(test_labels)


    #         # # print("test_images", test_images[4])
    #         # #####
    #         # test_batch_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    #         # test_batch_ds = test_batch_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
            
    #         # for batch_size_images, batch_size_labels in test_batch_ds:
    #             # test_step(batch_size_images, batch_size_labels)
    #         ####
    #         images = tf.cast(images, tf.float32)
    #         test_step(images, labels)
    #         test_bar.update(step)
        
    #     with val_summary_writer.as_default():
    #         tf.summary.scalar('loss', test_loss.result(), step=epoch+1)
    #         tf.summary.scalar('accuracy', test_accuracy.result()*100, step=epoch+1)

    #     print('Epoch: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'.format(epoch+1,train_loss.result(),
    #                         train_accuracy.result()*100, test_loss.result(),test_accuracy.result()*100))
        
    #     print("Spends time({}) in Epoch {}".format(epoch+1, time.perf_counter() - start))

        # if prev_test_accuracy >= test_accuracy.result():
        #     performance_lr_scheduling()
        # prev_test_accuracy = test_accuracy.result()