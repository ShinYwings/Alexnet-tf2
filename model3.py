import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras
import tensorflow.keras.backend as K 

def mAlexNet(INPUT_SHAPE, NUM_CLASSES):
    
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(INPUT_SHAPE, INPUT_SHAPE),
        tf.keras.layers.Conv2D(96, kernel_size=(11,11), input_shape = (INPUT_SHAPE, INPUT_SHAPE, 3),
                                            strides=(4,4), padding="valid", 
                                            activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(256,kernel_size=(5,5), strides=(1,1), padding="same",
                                            activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(384,kernel_size=(3,3), strides=(1,1), padding="same",
                                            activation='relu'),
        tf.keras.layers.Conv2D(384,kernel_size=(3,3), strides=(1,1), padding="same",
                                            activation='relu'),
        
        tf.keras.layers.Conv2D(256,kernel_size=(3,3), strides=(1,1), padding="same",
                                        activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

# class mAlexNet(tf.keras.Model):
#     def __init__(self, INPUT_SHAPE, NUM_CLASSES):

#         """ 
#         tf.keras.layers.Conv2D(
#             filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
#             dilation_rate=(1, 1), groups=1, activation=None, use_bias=True,
#             kernel_initializer='glorot_uniform', bias_initializer='zeros',
#             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#             kernel_constraint=None, bias_constraint=None, **kwargs
#         )
        
#         """
#         super(mAlexNet, self).__init__()
        
#         self.INPUT_SHAPE = INPUT_SHAPE
#         self.NUM_CLASSES = NUM_CLASSES

#         self.conv1 = tf.keras.layers.Conv2D(96, kernel_size=(11,11), input_shape = (self.INPUT_SHAPE, self.INPUT_SHAPE, 3),
#                                             strides=(4,4), padding="valid", 
#                                             activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(0))
#         self.conv2 = tf.keras.layers.Conv2D(256,kernel_size=(5,5), strides=(1,1), padding="same",
#                                             activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
#         self.conv3 = tf.keras.layers.Conv2D(384,kernel_size=(3,3), strides=(1,1), padding="same",
#                                             activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(0))
#         self.conv4 = tf.keras.layers.Conv2D(384,kernel_size=(3,3), strides=(1,1), padding="same",
#                                             activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
#         self.conv5 = tf.keras.layers.Conv2D(256,kernel_size=(3,3), strides=(1,1), padding="same",
#                                             activation='relu',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        
#         self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")
#         self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")
#         self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")

#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         self.bn3 = tf.keras.layers.BatchNormalization()

#         self.flatten = tf.keras.layers.Flatten()
#         self.fc1 = tf.keras.layers.Dense(4096,activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
#         self.dropout1 = tf.keras.layers.Dropout(0.2)
#         self.fc2 = tf.keras.layers.Dense(4096, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
#         self.dropout2 = tf.keras.layers.Dropout(0.2)
#         self.fc3 = tf.keras.layers.Dense(self.NUM_CLASSES, activation=tf.nn.softmax, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))

#         # self.mul1 = mulLayer(weight_init=0.2)
#         # self.mul2 = mulLayer(weight_init=0.2)
        
#     def call(self, x, training=False):
        
#         # x = inputs
#         # for layer in self.feature:
#         #     x = layer(x)

#         # return x
#         # assert type(training) is not bool, print("training must be a boolean type")
        
#         # 1st layer
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.bn1(x, training= training)
        
#         # 2nd layer
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.bn2(x, training= training)

#         # 3rd layer
#         x = self.conv3(x)
        
#         # 4th layer
#         x = self.conv4(x)
        
#         # 5th layer
#         x = self.conv5(x)
#         x = self.pool3(x)
#         x = self.bn3(x, training= training)

#         # fc layers
#         x = self.flatten(x)
#         x = self.fc1(x)
        
#         # if training:
#         x = self.dropout1(x, training= training)
#         x = self.fc2(x)
#         x = self.dropout2(x, training= training)
#         # else:
#         #     x = self.mul1(x)
#         #     x = self.fc2(x)
#         #     x = self.mul2(x)
            
#         return self.fc3(x)