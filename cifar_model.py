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

        self.conv1 = tf.keras.layers.Conv2D(96, kernel_size=(2,2), 
                                            strides=(2,2), padding="valid", 
                                            activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(0))
        self.conv2 = tf.keras.layers.Conv2D(96,kernel_size=(2,2), strides=(1,1), padding="same",
                                            activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.conv3 = tf.keras.layers.Conv2D(256,kernel_size=(2,2), strides=(1,1), padding="same",
                                            activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(0))
        self.conv4 = tf.keras.layers.Conv2D(384,kernel_size=(1,1), strides=(1,1), padding="same",
                                            activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.conv5 = tf.keras.layers.Conv2D(384,kernel_size=(1,1), strides=(1,1), padding="same",
                                            activation='relu',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding="valid")
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding="valid")
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding="valid")

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1024,activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(1024,activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.fc3 = tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))

    def call(self, x, training=None):
        
        # assert type(training) is not bool, print("training must be a boolean type")
        print("inputs is ", tf.shape(x))
        radius, alpha, beta, bias = self.LRN_INFO
        
        if training:
            # 1st layer
            inputs = self.conv1(x)
            lrn1 = tf.nn.local_response_normalization(inputs, depth_radius=radius,
                                                            alpha=alpha, beta=beta, bias=bias)
            mp1 = self.pool1(lrn1)
            
            # 2nd layer
            cnv2 = self.conv2(mp1)
            lrn2 = tf.nn.local_response_normalization(cnv2, depth_radius=radius,
                                                            alpha=alpha, beta=beta, bias=bias)
            mp2 = self.pool2(lrn2)

            # 3rd layer
            cnv3 = self.conv3(mp2)
            
            # 4th layer
            cnv4 = self.conv4(cnv3)
            
            # 5th layer
            cnv5 = self.conv5(cnv4)
            lrn3 = tf.nn.local_response_normalization(cnv5, depth_radius=radius,
                                                            alpha=alpha, beta=beta, bias=bias)
            mp3 = self.pool3(lrn3)

            # fc layers
            ft = self.flatten(mp3)
            fcl1 = self.fc1(ft)
            do1 = self.dropout1(fcl1)
            fcl2 = self.fc2(do1)
            do2 = self.dropout2(fcl2)
            
            return self.fc3(do2)

        else:   # test model
            # 1st layer
            inputs = self.conv1(x)
            lrn1 = tf.nn.local_response_normalization(inputs, depth_radius=radius,
                                                            alpha=alpha, beta=beta, bias=bias)
            mp1 = self.pool1(lrn1)
            
            # 2nd layer
            cnv2 = self.conv2(mp1)
            lrn2 = tf.nn.local_response_normalization(cnv2, depth_radius=radius,
                                                            alpha=alpha, beta=beta, bias=bias)
            mp2 = self.pool2(lrn2)

            # 3rd layer
            cnv3 = self.conv3(mp2)
            
            # 4th layer
            cnv4 = self.conv4(cnv3)
            
            # 5th layer
            cnv5 = self.conv5(cnv4)
            lrn3 = tf.nn.local_response_normalization(cnv5, depth_radius=radius,
                                                            alpha=alpha, beta=beta, bias=bias)
            mp3 = self.pool3(lrn3)

            # fc layers
            ft = self.flatten(mp3)
            fcl1 = self.fc1(ft)
            fcl2 = self.fc2(fcl1)

            return self.fc3(fcl2 / 2.0 )
