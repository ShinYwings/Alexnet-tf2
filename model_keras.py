import tensorflow as tf
from tensorflow.keras import Model

class AlexNet(Model):
    def __init__(self):

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
        super(AlexNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(96, input_shape=(227,227,3) ,kernel_size=(11,11), 
                                            strides=(4,4), padding="valid", 
                                            activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(96,kernel_size=(5,5), strides=(1,1), padding="same", 
                                            activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(256,kernel_size=(3,3), strides=(1,1), padding="same",
                                            activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(384,kernel_size=(3,3), strides=(1,1), padding="same",
                                            activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(384,kernel_size=(3,3), strides=(1,1), padding="same",
                                            activation='relu')
        
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096,activation='relu')
        self.fc2 = tf.keras.layers.Dense(4096,activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
    
    def call(self, x, lrn_info, NUM_CLASSES):
        
        radius, alpha, beta, bias = lrn_info

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
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

        return x
