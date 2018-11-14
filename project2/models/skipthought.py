from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from utils.constants import CONST


class Skipthought(BaseModel):
    def __init__(self, args, debugger):
        super(Skipthought, self).__init__(args, debugger)
        self.build_model()

    def build_model(self):
        self.debugger.debug('==============Building model==============')
        # create word embeddings for the words in the sentences.
        self.sentences = tf.placeholder(
            tf.int32, [self.batch_size, 5, self.args.sentence_length], name='sentences')
        self.training = tf.placeholder_with_default(False, shape=(), name='training')
        with tf.variable_scope('encodings'):
            self.encodings = tf.placeholder(
                tf.float32, shape=[self.batch_size, 5, CONST['SKIP_SIZE']])

        with tf.variable_scope('lstm_cell'):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(CONST['LSTM_SIZE'])
            initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        with tf.variable_scope('LSTM'):
            rnn_outputs, _ = tf.nn.dynamic_rnn(
                lstm_cell,
                self.encodings,
                initial_state=initial_state,
                dtype=tf.float32)

        attention_size = 50
        with tf.variable_scope('attention'):
            W_attention = tf.get_variable(
                'W_attention', shape=[CONST['LSTM_SIZE'], attention_size], initializer=xavier_initializer(), )
            b_attention = tf.get_variable(
                'b_attention', shape=[attention_size], initializer=xavier_initializer())
            u_attention = tf.get_variable(
                'u_attention', shape=[attention_size], initializer=xavier_initializer())
            v = tf.tanh(tf.tensordot(rnn_outputs, W_attention, axes=1) + b_attention)

            vu = tf.tensordot(v, u_attention, axes=1, name='vu')
            alphas = tf.nn.softmax(vu, name='alphas')

            attention_output = tf.reduce_sum(rnn_outputs * tf.expand_dims(alphas, -1), 1)

        with tf.variable_scope('dense'):
            self.debugger.debug('Creating dense layers')

            dense_1 = tf.layers.dense(
                attention_output, units=CONST['LSTM_SIZE'], activation=tf.nn.relu)

            self.logits = tf.layers.dense(dense_1, 2)

        self.labels = tf.placeholder(tf.int64, shape=[self.batch_size], name='Y')
        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels))
        self.probabilities = tf.nn.softmax(self.logits)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.loss_summary = tf.summary.scalar('sigmoid_mean_loss', self.loss)

        self.summaries = tf.summary.merge_all()

        self.debugger.debug('==============Model built==============')
