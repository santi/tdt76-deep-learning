import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from .model import Model
from utils.constants import CONST




class VanillaLSTM(Model):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.build_model()

        self.note_size = 20

    def build_model(self):
        self.logger.log('==============Building model==============')
        # create word embeddings for the words in the sentences.
        self.training = tf.placeholder_with_default(False, shape=(), name='training')

        self.context_length = 5
        self.note_size = 3

        self.X = tf.placeholder(tf.float32, [self.batch_size, self.context_length, self.note_size], name='X')


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
                outputs, units=self.note_size, activation=tf.nn.sigmoid)
        
        self.Y = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.context_length, self.note_size],
            name='Y')
        
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y))
            self.loss_summary = tf.summary.scalar('sigmoid_mean_loss', self.loss)
        """
        self.probabilities = tf.nn.softmax(self.logits)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        """
        
        self.summaries = tf.summary.merge_all()
        self.logger.log('==============Model built==============')
