import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras

class mulLayer(tf.keras.layers.Layer):
    def __init__(self, weight_init="weight_init"):
        super(mulLayer, self).__init__()
        self.weight_init = weight_init

    def call(self, input):
        return tf.scalar_mul(self.weight_init,input)

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
        super(mAlexNet, self).__init__()
        self._INPUT_SHAPE = INPUT_SHAPE
        self.NUM_CLASSES = NUM_CLASSES
        self.radius, self.alpha, self.beta, self.bias = LRN_INFO

        self.conv1 = tf.keras.layers.Conv2D(96, kernel_size=(11,11), input_shape = (self._INPUT_SHAPE,self._INPUT_SHAPE, 3),
                                            strides=(4,4), padding="valid", 
                                            activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(0))
        self.conv2 = tf.keras.layers.Conv2D(256,kernel_size=(5,5), strides=(1,1), padding="same",
                                            activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.conv3 = tf.keras.layers.Conv2D(384,kernel_size=(3,3), strides=(1,1), padding="same",
                                            activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(0))
        self.conv4 = tf.keras.layers.Conv2D(384,kernel_size=(3,3), strides=(1,1), padding="same",
                                            activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.conv5 = tf.keras.layers.Conv2D(256,kernel_size=(3,3), strides=(1,1), padding="same",
                                            activation='relu',kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")

        self.lrn1 = lambda inputs: tf.nn.local_response_normalization(inputs, depth_radius=self.radius,
                                                            alpha=self.alpha, beta=self.beta, bias=self.bias)
        self.lrn2 = lambda inputs : tf.nn.local_response_normalization(inputs, depth_radius=self.radius,
                                                            alpha=self.alpha, beta=self.beta, bias=self.bias)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096,activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.fc3 = tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(1))

        self.mul1 = mulLayer(weight_init=0.5)
        self.mul2 = mulLayer(weight_init=0.5)

    def call(self, x, training=None):
        
        # assert type(training) is not bool, print("training must be a boolean type")
        
        if training:

            # 1st layer
            inputs = self.conv1(x)
            lrn1 = self.lrn1(inputs)
            mp1 = self.pool1(lrn1)
            
            # 2nd layer
            cnv2 = self.conv2(mp1)
            lrn2 = self.lrn2(cnv2)
            mp2 = self.pool2(lrn2)

            # 3rd layer
            cnv3 = self.conv3(mp2)
            
            # 4th layer
            cnv4 = self.conv4(cnv3)
            
            # 5th layer
            cnv5 = self.conv5(cnv4)
            mp3 = self.pool3(cnv5)

            # fc layers
            ft = self.flatten(mp3)
            fcl1 = self.fc1(ft)
            do1 = self.dropout1(fcl1)
            fcl2 = self.fc2(do1)
            do2 = self.dropout2(fcl2)
            fcl3 = self.fc3(do2)

            return fcl3

        else:   # test model
            # 1st layer
            inputs = self.conv1(x)
            lrn1 = self.lrn1(inputs)
            mp1 = self.pool1(lrn1)
            
            # 2nd layer
            cnv2 = self.conv2(mp1)
            lrn2 = self.lrn2(cnv2)
            mp2 = self.pool2(lrn2)

            # 3rd layer
            cnv3 = self.conv3(mp2)
            
            # 4th layer
            cnv4 = self.conv4(cnv3)
            
            # 5th layer
            cnv5 = self.conv5(cnv4)
            mp3 = self.pool3(cnv5)

            # fc layers
            ft = self.flatten(mp3)
            fcl1 = self.fc1(ft)
            mul1 = self.mul1(fcl1)
            fcl2 = self.fc2(mul1)
            # multiply their outputs by 0.5
            mul2 = self.mul2(fcl2)
            fcl3 = self.fc3(mul2)

            return fcl3