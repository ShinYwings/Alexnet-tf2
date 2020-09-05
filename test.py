import tensorflow as tf 
import numpy as np
from datetime import datetime
from threading import Thread

class test():

    def __init__(self,model="model", optimizer="optimizer", 
                    dataset_test="dataset_test", labels_test="labels_test",
                        learning_rate="learning_rate", log_freq=10, 
                        epochs="epochs", momentum = "momentum",
                            display_step="display_step"):
        self.model = model
        self.optimizer = optimizer
        self.dataset_test = dataset_test
        self.labels_test = labels_test
        self.learning_rate = learning_rate
        self.log_freq = log_freq
        self.epochs = epochs
        self.momentum = momentum
        self.display_step = display_step
        
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
        
        #####################

        pred = self.model.classifier(x, 1.0)    

        avg_prediction = tf.div(tf.reduce_sum(pred,0), k_patches)

        top1_correct = tf.equal(tf.argmax(avg_prediction,0), tf.argmax(y,1))
        top1_accuracy = tf.reduce_mean(tf.cast(top1_correct, tf.float32))

        top5_correct = tf.nn.in_top_k(tf.stack([avg_prediction]), tf.argmax(y,1), k=5)
        top5_accuracy = tf.reduce_mean(tf.cast(top5_correct, tf.float32))

        total_top1_accuracy = 0.
        total_top5_accuracy = 0.

        for i in range(self.dataset_test):

            image_patches = tu.read_k_patches(self.dataset_test, k_patches)
            label = self.labels_test

            top1_a, top5_a = top1_accuracy(self.dataset_test, self.labels_test)
            total_top1_accuracy += top1_a
            total_top5_accuracy += top5_a

            if i % display_step == 0:
                print("Examples done : {:05d}/{} ---- Top-1: {:.4f} -- Top-5: {:.4f}".format(i+1, self.dataset_test, total_top1_accuracy / (i+1), total_top5_accuracy/(i+1)))

            print( "--- Final accuracy ---")
            print("Top-1: {:.4f} -- Top-5 error rate: {:.4f}".format(1-(total_top1_accuracy/self.dataset_test), 1-(total_top5_accuracy/self.dataset_test)))