import tensorflow as tf
import numpy as np
from datetime import datetime
from data_generator import unpickle


def Alexnet(X, DROP_PROB="DROP_PROB", NUM_CLASSES="NUM_CLASSES", FC_LAYERS="FC_LAYERS", weight_path='DEFAULT'):

    """Create the graph of the AlexNet model.
    Args:
        x: Placeholder for the input tensor.
        dropout_prob: Dropout probability.
        num_classes: Number of classes in the dataset.
        fc_layer: List of names of the layer, that get trained from
            scratch
        weights_path: Complete path to the pretrained weight file, if it
            isn't in the same folder as this code
    """
    if weight_path == 'DEFAULT':
        WEIGHTS_PATH = 'bvlc_alexnet.npy'
    else:
        WEIGHTS_PATH = weight_path

    # 1st layer: conv -> lrn -> pool
    conv1 = conv(x=X, filter_height=11, filter_width=11, 
                    num_filters=96, stride_y=4, stride_x=4, 
                        padding='VAILD', name='conv1')
    norm1 = lrn(conv1, radius=5, bias=1, alpha=1e-4, beta=0.75, name='norm1')
    pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
    
    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool ~with 2 groups~
    conv2 = conv(pool1, 5, 5, 256, 1, 1, name='conv2')
    norm2 = lrn(conv2, radius=5, bias=1, alpha=1e-4, beta=0.75, name='norm2')
    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
    
    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, name='conv4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
    norm5 = lrn(conv5, radius=5, bias=1, alpha=1e-4, beta=0.75, name="norm5")
    pool5 = max_pool(norm5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 6*6*256])
    fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
    dropout6 = dropout(fc6, DROP_PROB)

    # 7th Layer: FC (w ReLu) -> Dropout
    fc7 = fc(dropout6, 4096, 4096, name='fc7')
    dropout7 = dropout(fc7, DROP_PROB)

    # 8th Layer: FC and return unscaled activations
    fc8 = fc(dropout7, 4096, NUM_CLASSES, relu=False, name='fc8')

    cost = tf.nn.softmax(fc8, name="cost")

    return cost

# def load_initial_weights(session):
#     """Load weights from file into network.
#     As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#     come as a dict of lists (e.g. weights['conv1'] is a list) and not as
#     dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
#     'biases') we need a special load function
#     """
#     # Load the weights into memory
#     weights_dict = np.load(WEIGHTS_PATH, encoding='bytes').item()

#     # Loop over all layer names stored in the weights dict
#     for op_name in weights_dict:

#         # Check if layer should be trained from scratch
#         if op_name not in FC_LAYERS:

#             # Assign weights/biases to their corresponding tf variable
#             for data in weights_dict[op_name]:

#                 # Biases
#                 if len(data.shape) == 1:
#                     var = tf.Variable(name='biases', trainable=False)
#                     var.assign(data)

#                 # Weights
#                 else:
#                     var = tf.Variable(name='weights', trainable=False)
#                     var.assign(data)


def conv(x="x", filter_height="filter_height", filter_width="filter_width",
            num_filters="num_filters", stride_y="stride_y", stride_x="stride_x", 
                name="name", padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    filters = tf.zeros(shape=[filter_height,filter_width,input_channels, num_filters])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    # Create tf variables for the weights and biases of the conv layer
    weights = tf.Variabe(name='weights', shape=[filters])
    biases = tf.Variable(name='biases', shape=[num_filters])

    conv = convolve(x, weights)

    """
    Multiple GPUs 할 때 싸용
    """
    # # In the cases of multiple groups, split inputs & weights and
    # else:
    #     # Split input and weights and convolve them separately
    #     input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
    #     weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
    #     output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

    #     # Concat the convolved output together again
    #     conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=name)

    return relu


def fc(x, num_in, num_out, name, relu=True):

    """Create a fully connected layer."""
    # Create tf variables for the weights and biases
    weights = tf.Variable(name='weights', shape=tf.TensorShape([num_in, num_out]), trainable=True)
    biases = tf.Variable(name='biases', shape=tf.TensorShape[num_out], trainable=True)

    # Matrix multiply weights and inputs and add bias
    @tf.function
    def forward(x):
        return  tf.matmul(x, weights) + biases

    act = forward(x)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    """Create a max pooling layer."""
    # 여기서 0, 3번째의 1의 의미: 첫번째 1은 batch에 대한 윈도우 크기, 마지막은 채널에 대한 윈도우 크기
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=2.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)
    #depth_radius = 주변에 몇개까지 할 것인지... 5이면 자신 기준 [-2,2]
    # bias: hyperparameter k in the paper

def dropout(x, DROP_PROP):

    """Create a dropout layer."""
    return tf.nn.dropout(x, DROP_PROP)

"""
tf.metrics

summary로 기록할 데이터를 수집
"""

# def test(model, test_x, test_y, step_num):
#     loss = loss_fn(model(test_x), test_y)
#     tf.summary.scalar('loss', loss, step=step_num)


# test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')



# with test_summary_writer.as_default():
#     test(model, test_x, test_y, optimizer.iterations)

"""
Watch the summaries in Tensorboard

type "tensorboard --logdir /tmp/summaries" in cli of conda prompt

"""

