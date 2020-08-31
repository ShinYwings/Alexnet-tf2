import tensorflow as tf 
import numpy as np

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

class ImageDataGenerator(object):

    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True, buffer_size=1000):

        """
        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.
        """

        self.txt_file = txt_file
        self.num_classes = num_classes
        self.__read_txt_file()

        self.data_size = len(self.labels)

        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        if mode == 'training':
            data = data.map(self._parse_function_train, num_threads=8, output_buffer_size=100*batch_size)
        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_threads=8,
                output_buffer_size=100*batch_size)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        if shuffle:
            data = data.shuffle(buffer_size= buffer_size)

        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self):

        self.img_paths = []
        self.labels = []
    with open(self.txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            self.img_paths.append(items[0])
            self.labels.append(int(items[1]))

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot