import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras

class mAlexNet(Model):
    def __init__(self, INPUT_SHAPE, LRN_INFO, NUM_CLASSES):

        """ 
        tf.keras.layers.Conv2D(
            filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
            dilation_rate=(1, 1), groups=1, activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None, **kwargs
        )
        
        """

        # data_format='channels_first'
        super(mAlexNet, self).__init__()
        self._INPUT_SHAPE = INPUT_SHAPE
        self.LRN_INFO = LRN_INFO
        self.NUM_CLASSES = NUM_CLASSES
        
        self.conv1 = tf.keras.layers.Conv2D(96, kernel_size=(40,40), 
                                            strides=(4,4), padding="valid", 
                                            activation='relu',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(0))
        self.conv2 = tf.keras.layers.Conv2D(96,kernel_size=(5,5), strides=(1,1), padding="same", 
                                            activation='relu',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.conv3 = tf.keras.layers.Conv2D(256,kernel_size=(3,3), strides=(1,1), padding="same",
                                            activation='relu',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(0))
        self.conv4 = tf.keras.layers.Conv2D(384,kernel_size=(3,3), strides=(1,1), padding="same",
                                            activation='relu',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.conv5 = tf.keras.layers.Conv2D(384,kernel_size=(3,3), strides=(1,1), padding="same",
                                            activation='relu',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096,activation='relu',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.fc2 = tf.keras.layers.Dense(4096,activation='relu',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.fc3 = tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))

    def call(self, x):
        
        radius, alpha, beta, bias = self.LRN_INFO
        
        # 1st layer
        x = self.conv1(x)
        x = tf.nn.local_response_normalization(x, depth_radius=radius,
                                                        alpha=alpha, beta=beta, bias=bias)
        x = self.pool1(x)

        # 2nd layer
        x = self.conv2(x)
        x = tf.nn.local_response_normalization(x, depth_radius=radius,
                                                        alpha=alpha, beta=beta, bias=bias)
        x = self.pool2(x)

        # 3rd layer
        x = self.conv3(x)
        
        # 4th layer
        x = self.conv4(x)
        
        # 5th layer
        x = self.conv5(x)
        x = tf.nn.local_response_normalization(x, depth_radius=radius,
                                                        alpha=alpha, beta=beta, bias=bias)
        x = self.pool3(x)

        # fc layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x/2)
        
        return x
