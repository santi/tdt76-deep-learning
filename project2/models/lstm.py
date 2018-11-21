import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from .model import Model
from utils.constants import CONST




class VanillaLSTM(Model):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.build_model()

    def build_model(self):
        self.logger.log('==============Building model==============')
        # create word embeddings for the words in the sentences.
        self.training = tf.placeholder_with_default(False, shape=(), name='training')

        self.X = tf.placeholder(tf.float32, [self.batch_size, CONST['SEQUENCE_LENGTH'], CONST['NOTE_LENGTH']], name='X')


        with tf.variable_scope('lstm_cell'):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(CONST['LSTM_SIZE'])
            initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        with tf.variable_scope('LSTM'):
            outputs, _ = tf.nn.dynamic_rnn(
                lstm_cell,
                inputs=self.X,
                initial_state=initial_state,
                dtype=tf.float32)
        
        with tf.variable_scope('dense'):
            self.logits = tf.layers.dense(
                outputs, units=CONST['NOTE_LENGTH'], activation=tf.nn.sigmoid)
            print(self.logits.shape)
        
        self.Y = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, CONST['SEQUENCE_LENGTH'], CONST['NOTE_LENGTH']),
            name='Y')
        
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(
                logits=self.logits, targets=self.Y, pos_weight=1.0/50)) / (self.args.batch_size * CONST['SEQUENCE_LENGTH'] * CONST['NOTE_LENGTH'])
            self.loss_summary = tf.summary.scalar('sigmoid_mean_loss', self.loss)
        """
        self.probabilities = tf.nn.softmax(self.logits)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        """

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer()
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.train_step = self.optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step_tensor)
        
        self.summaries = tf.summary.merge_all()
        self.logger.log('==============Model built==============')
