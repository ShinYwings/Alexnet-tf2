import tensorflow as tf
import numpy as np
from datetime import datetime
from threading import Thread

class train():

    def __init__(self,model="model", optimizer="optimizer", 
                    dataset_train="dataset_train", dataset_val="dataset_val", 
                        learning_rate="learning_rate", log_freq=10, epochs="epochs",
                            momentum = "momentum"):
        
        self.model = model
        self.optimizer = optimizer
        self.dataset_train = dataset_train
        self.dataset_val= dataset_val
        self.learning_rate = learning_rate
        self.log_freq = log_freq
        self.epochs = epochs
        self.momentum = momentum
        
    def start(self):
        avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
        
        for images, labels in dataset:
            
            with tf.GradientTape() as tape:
                loss = train_step(model, optimizer, images, labels)
                avg_loss.update_state(loss)

                if tf.equal(optimizer.iterations % log_freq, 0):
                    tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
                    avg_loss.reset_states()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        
        with tf.device('/gpu:1'):

            @tf.function
            def cross_entropy(logits="logits", labels="labels", name="cross-entropy"):

                return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name=name))
            
            @tf.function
            def l2_loss(lmbda="lambda", wx_plus_b= "wx_plus_b", name="ls_loss"):
                l2_loss_value = tf.reduce_sum(lmbda*tf.stack([tf.nn.l2_loss(wx_plus_b) for wx_plus_b in tf.get_collection('wx_plus_b')]))
                tf.summary.scalar('l2_loss', l2_loss_value)

            @tf.function
            def loss(cross_entropy="cross_entropy", l2_loss_value="l2_loss_value"):
                loss_value = cross_entropy + l2_loss_value
                tf.summary.scalar('loss', loss_value)
                return loss_value

            @tf.function
            def accuracy(pred="pred", y_b="y_b"):
                correct = tf.equal(tf.argmax(pred,1) , tf.argmax(y_b,1))
                accuracy_value = tf.reduce_mean(tf.cast(correct, tf.float32))
                tf.summary.scalar('accuracy', accuracy_value)

                return accuracy_value

            global_step = tf.Variable(0, trainable=False)
            epoch = tf.div(global_step, num_batches)

            @tf.function
            def optimizer(learning_rate="learning_rate", loss="loss", global_step="global_step", momentum="momentum"):
                
                # SGD 로 바꾸기
                return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

            # 이거 v2에서 필요없다라고 한거 같음
            merged = tf.summary.merge_all()

            coord = tf.train.Coordinator()

            @tf.function
            def enqueue_batches():
                while not coord.should_stop():
                    im, l = tu.read_batch(batch_size, train_img_path, wnid_labels)

                    sess.run(enqueue_op, feed_dict={x: im, y:l})

            num_threads = threads_number

            for i in range(num_threads):
                i = Thread(target=enqueue_batches) #prefetch???
                t.setDaemon(True)
                t.start()

            train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')

            with train_summary_writer.as_default():
                train(model, optimizer, dataset)

            start_time = datetime.time()

            for e in range(epoch, epochs):

                for i in range(num_batches):

                    optimizer(learning_rate = learning_rate, loss = loss,
                                global_step=global_step, momentum= momentum)
                    
                    step = global_step

                    #Weight Decay
                    if step == 170000 or step == 350000:
                        learning_rate /= 10
            
                    if step % display_step == 0:
                        temp_time= datetime.time()
                        c = loss(cross_entropy=cross_entropy, l2_loss_value="l2_loss_value")
                        a = accuracy(pred=pred, y_b=y_b)

                    print("time: ", temp_time-start_time, 'Epoch: {:03d} Step/Batch: {:09d}--- Loss: {:.7f} Training accuracy: {:.4f}'.format(e,step,c,a))

                    if step % test_step == 0:
                        val_im, val_cls = tu.read_validation_batch(batch_size, val_images, val_labels)

                        v_a = accuracy(pred=val_im, y_b=val_cls)

                        # intermediate time
                        int_time = datetime.time()

                        print("Elapsed time: {}".format(tu.format_time(int_time-start_time)))
                        print("Validation accuracy: {:0.4f".format(v_a))

            end_time = datetime.time()
            print("Elapsed time: {}".format(tu.format_time(end_time - start_time)))

            coord.request_stop()



