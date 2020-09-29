import tensorflow as tf 

class mulLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs="num_outputs", weight_init="weight_init"):
        super(mulLayer, self).__init__()
        self.num_outputs = num_outputs
        self.weight_init = weight_init


    def build(self, input_shape):
        self.kernel = tf.Variable(initial_value= tf.constant(self.weight_init,
                                        shape=[int(input_shape[-1]),
                                         self.num_outputs], dtype=tf.float32),
                                         trainable=False)
    def call(self, input):
        return tf.scalar_mul(input, self.kernel)

if __name__ == "__main__":
    a  = mulLayer(num_outputs=2, weight_init=0.5)

    b = tf.Variable(initial_value= tf.constant(2.0,
                                        shape=[1,3], dtype=tf.float32),
                                         trainable=False)

    
    tf.print(b)
    tf.print(tf.scalar_mul(0.5,b))