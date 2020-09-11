import tensorflow as tf 
import numpy as np
import os
import sys 
import model_keras as model
import test_model as tmodel
# TODO change to data_Generator when TFrecord will be built in
import time
from datetime import datetime as dt
from matplotlib import pyplot as plt

# Hyper parameters
# TODO : argparse?
LEARNING_RATE = 0.01
NUM_EPOCHS = 90
NUM_CLASSES = 10    # CIFAR-10
MOMENTUM = 0.9 # SGD + MOMENTUM
BATCH_SIZE = 128
DATASET_DIR = r"D:\cifar-10-batches-py"
TRAIN_TFRECORD_DIR = r"D:\cifar-10-size-256\train_tfrecord"
TEST_TFRECORD_DIR = r"D:\cifar-10-size-256\test_tfrecord"
LRN_INFO = (5, 1e-4, 0.75, 2) # radius, alpha, beta, bias
INPUT_IMAGE_SIZE = 256 #WIDTH, HEIGHT  #x를 224x224로 해보고 그다음에 227x227로 바꾸기
WEIGHT_DECAY = 5e-4
# TODO optimizer (weight decay & lr/10 heuristic 방법) 추가

# Fixed
DROUPUT_PROP = 0.5
ENCODING_STYLE = "utf-8"
AUTO = tf.data.experimental.AUTOTUNE
# How often we want to write the tf.summary data to disk
DISPLAY_STEP = 20

def get_run_logdir(root_logdir):
    run_id = dt.now().strftime("run_%Y_%m_%d-%H_%M_%S")
    
    return os.path.join(root_logdir, run_id)

def preprocess_images(ds):
    print("image resizing...")

    global INPUT_IMAGE_SIZE
    refined_ds = list()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            with tf.device("/gpu:1"):
        
                for image, label in ds:
                    """N(0,1)로 norm ( 정확히는 norm은 아닌데 0~255로 fix되어 있으니까 norm이라 봐도 무방"""    
                    image = tf.image.per_image_standardization(image)
                    """resize images from 32X32 to INPUT_IMAGE_SIZE x INPUT_IMAGE_SIZE (alexnet standard)"""
                    image2 = tf.image.resize(image, (INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE))
                    refined_ds.append((image2, label))
        
        except RuntimeError as e:
            # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
            print(e)
    
    return refined_ds

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

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
    
    train_tfrecord_list = tf.io.matching_files(os.path.join(TRAIN_TFRECORD_DIR, '*.tfrecord'))
    test_tfrecord_list = tf.io.matching_files(os.path.join(TEST_TFRECORD_DIR, '*.tfrecord'))

    print("train_tfrecord_list len", len(train_tfrecord_list))
    print("test_tfrecord_list len", len(test_tfrecord_list))

    train_ds = tf.data.TFRecordDataset(train_tfrecord_list, num_parallel_reads=AUTO)
    test_ds = tf.data.TFRecordDataset(test_tfrecord_list, num_parallel_reads=AUTO)
    mapped_tr_ds = train_ds.map(_parse_function)
    mapped_ts_ds = test_ds.map(_parse_function)
    # train_ds_size = tf.data.experimental.cardinality(parsed_tr_ds).numpy()
    parsed_tr_ds = mapped_tr_ds.shuffle(buffer_size=50000).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    
    # test_ds_size = tf.data.experimental.cardinality(parsed_ts_ds).numpy()
    parsed_ts_ds = mapped_ts_ds.shuffle(buffer_size=10000).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    
    # train_batch = iter(parsed_tr_ds.unbatch().batch(128)) # 이거 지울까 말까 고민중
    # test_batch = iter(parsed_ts_ds.unbatch().batch(128)) # 이거 지울까 말까 고민중

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
    
    # val_ds_size = tf.data.experimental.cardinality(val_ds).numpy()
    # print("Training data size: ", train_ds_size)
    # print("Test data size: ", test_ds_size)
    # print("Val data size: ", val_ds_size)

    # with tf.device('/gpu:1'):
    #     resize_and_rescale = tf.keras.Sequential([
    #                 tf.keras.layers.experimental.preprocessing.Resizing(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
    #                 tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    #             ])
    # Data Preprocessing STRATEGY 
    # map(train_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    
    # val_ds = (val_ds.map(preprocess_images, num_parallel_calls=5)
    #                         .shuffle(buffer_size=val_ds_size)
    #                         .batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(1))

    """check images are all right""" 
    
    # plt.figure(figsize=(20,20))

    # for i, (image,_) in enumerate(train_ds.take(5)):
    #     ax = plt.subplot(5,5,i+1)
    #     plt.imshow(image[i])
    #     plt.axis('off')
    # plt.show()

    _model = model.mAlexNet(INPUT_IMAGE_SIZE, LRN_INFO, NUM_CLASSES)
    test_model = tmodel.mAlexNet(INPUT_IMAGE_SIZE, LRN_INFO, NUM_CLASSES)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=False)
    
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

    get_run_logdir: return the location of the exact directory that is named
                    according to the current time the training phase starts
    """

    root_logdir = os.path.join(filewriter_path, "logs\\fit\\")

    run_logdir = get_run_logdir(root_logdir)

    """
    Training and Results

    To train the network, we have to compile it.

    Compilation processes
        - Loss function
        - Optimization Algorithm
        - Learning Rate
    """

    summary_writer = tf.summary.create_file_writer(run_logdir)
    
    print('tensorboard --logdir={}'.format(run_logdir))

    with tf.device('/gpu:1'):
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = _model(images)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, _model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, _model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)

        @tf.function
        def test_step(images, labels):
            predictions = test_model(images)
            t_loss = loss_object(labels, predictions)
            test_loss(t_loss)
            test_accuracy(labels, predictions)
    
    train_iter = iter(parsed_tr_ds)
    test_iter = iter(parsed_ts_ds)
    print("시작")
    for epoch in range(NUM_EPOCHS):
        start = time.perf_counter()
        for tb in train_iter:
            
            raw_image= tb['image']
            raw_labels= tb['label'].numpy()
            
            images = list()
            for i in range(0,BATCH_SIZE):
                image = tf.image.decode_jpeg(raw_image[i], channels=3).numpy()
                tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
                norm_image = tf.image.per_image_standardization(tensor_image)
                images.append(norm_image)
                
                # labels.append(np.array(label))
                # image = tf.array(image[np.newaxis,...], dtype=tf.int32)
                # label = np.array(label[np.newaxis,...], dtype=tf.int32)
            # image = [tf.image.decode_jpeg(i, channels=3) for i in tb['image']]
            # label = [j for j in tb['label']]
            images_list = tf.convert_to_tensor(images, dtype=tf.float32)
            labels = tf.convert_to_tensor(raw_labels, dtype=tf.int64)
            
            # print("images", tf.shape(images))
            # print("labels", tf.shape(labels))
            # print("images size", np.shape(images))
            # print("images:",images)
        # for images, labels in parsed_tr_ds:
        #     print("images size", np.shape(images))
        #     print("images:",images)
            
            train_step(images_list, labels)
            
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch+1)
            tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch+1)

        for tb in test_iter:
            raw_image= tb['image']
            raw_labels= tb['label'].numpy()
            images = list()
            for i in range(0,BATCH_SIZE):
                image = tf.image.decode_jpeg(raw_image[i], channels=3).numpy()
                tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
                norm_image = tf.image.per_image_standardization(tensor_image)
                images.append(norm_image)
            images_list = tf.convert_to_tensor(images, dtype=tf.float32)
            labels = tf.convert_to_tensor(raw_labels, dtype=tf.int32)
        # for test_images, test_labels in parsed_ts_ds:
            test_step(images_list, labels)
            # image = [tf.image.decode_jpeg(i, channels=3) for i in tb['image']]
            # label = [j for j in tb['label']]
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
    # _iterator = iter(train_ds)

    # batch = _iterator.get_next()

    # nesterov: SGD momentum은 현재 그래디언트에다가 모멘텀을 곱한 가중치를 더한 가중치를 다음 가중치를 업데이트를 해줌 
    # 반면, nesterov는 
    # sgd = tf.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=False)
    # train(alexnet, optimizer= sgd, dataset_train=train_ds, dataset_val=val_ds, epochs=NUM_EPOCHS)
    # model.compile(optimizer=sgd, loss=loss_fn)
    # model.fit(dataset)
    # keras.Model.fit(dataset)