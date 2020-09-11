import tensorflow as tf
import numpy as np
from datetime import datetime

def train(model="model", optimizer="optimizer", 
                    train_images="train_images", train_labels="train_labels", 
                        LEARNING_RATE="LEARNING_RATE", DISPLAY_STEP="DISPLAY_STEP", epochs="epochs",
                            MOMENTUM = "MOMENTUM", BATCH_SIZE = "BATH_SIZE", LRN_INFO="LRN_INFO",
                                NUM_CLASSES= "NUM_CLASSES"):
    

    train_ds = tf.data.Dataset.from_tensor_slices((train_images[:], train_labels[:]))

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()

    print("Training data size: ", train_ds_size)

    train_ds = train_ds.shuffle(buffer_size=train_ds_size).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(1)

    _model = model.AlexNet()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=False)
    
    # 모델의 손실과 성능을 측정할 지표, 에포크가 진행되는 동안 수집된 측정 지표를 바탕으로 결과 출력
    train_loss = tf.keras.metrics.Mean(name= 'train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    root_logdir = os.path.join(filewriter_path, "logs\\fit\\")

    run_logdir = get_run_logdir(root_logdir)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

    summary_writer = tf.summary.create_file_writer('/tmp/summaries')

    with tf.device('/gpu:1'):
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = _model(images, LRN_INFO, NUM_CLASSES)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, _model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, _model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', 0.1, step=DISPLAY_STEP)
    
    @tf.function
    def test_step(images, labels):
        predictions = _model(images, LRN_INFO, NUM_CLASSES)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)
        with summary_writer.as_default():
            tf.summary.scalar('test_loss', 0.1, step=DISPLAY_STEP)
    
    for epoch in range(NUM_EPOCHS):
        for images, labels in train_ds:
            train_step(images, labels)
            print("트레이닝 중...")

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
        
        print('에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'.format(epoch+1,train_loss.result(),
                            train_accuracy.result()*100, test_loss.result(),test_accuracy.result()*100))
        summary_writer.flush()