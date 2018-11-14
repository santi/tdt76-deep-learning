import tensorflow as tf

class Model:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.batch_size = args.batch_size
        # init the epoch and global step counters
        self.init_step_counters()


    def init_step_counters(self):
        with tf.variable_scope('counters'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='current_epoch')
            self.increment_cur_epoch_tensor = tf.assign(
                self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')


    def build_model(self):
        raise NotImplementedError("Models must override build_model method")
