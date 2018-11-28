import tensorflow as tf
import sys

class Model:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.batch_size = args.batch_size
        # init the epoch and global step counters
        self.init_state_tensors()


    def init_state_tensors(self):
        with tf.variable_scope('counters'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='current_epoch')
            self.increment_cur_epoch_tensor = tf.assign(
                self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        with tf.variable_scope('validation_score'):
            self.validation_score = tf.Variable(sys.float_info.max, trainable=False, name='validation_score')
            self.set_validation_score = tf.placeholder(tf.float32, name='set_validation_score')
            self.assign_validation_score = tf.assign(
                self.validation_score, self.set_validation_score)
        


    def build_model(self):
        raise NotImplementedError("Models must override build_model method")
