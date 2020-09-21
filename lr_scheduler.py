import numpy as np
import tensorflow as tf
# val error rate 가 stop improving 1/10

class lr_func(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, learning_rate = "learning_rate"):
        super(lr_func, self).__init__()
        
        self.lr = learning_rate
        self.prev_loss = 0
        self.loss = 0

    def __call__(self, step):

        if step is 0:
            learning_rate = 0.1

        # 종료전 3배 감소
        elif step is 128:

            return tf.divide(learning_rate, 3.0)
        
        if np.isclose(prev_loss, loss):
            
            return tf.divide(learning_rate, 10.0)
        
        return learning_rate
        # return super().__call__(step)

    def from_config(cls, config):
        return super().from_config(config)

    def get_config(self):
        return super().get_config()