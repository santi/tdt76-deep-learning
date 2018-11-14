import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from .model import Model
from utils.constants import CONST


class VanillaLSTM(Model):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.build_model()

    def build_model(self):
        # TODO: Change method to allow testing of model, not just training

        self.logger.debug('==============Building model==============')
        # create word embeddings for the words in the sentences.
        self.training = tf.placeholder_with_default(False, shape=(), name='training')

        with tf.variable_scope('embeddings'):
            pass

        with tf.variable_scope('lstm_cell'):
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(
                CONST['LSTM_SIZE'], state_is_tuple=True)

        lstm_outputs = []
        for i, embedding in enumerate(embeddings):
            lstm_outputs.append(
                self.sentence_lstm(
                    embedding,
                    reuse=(i != 0)
                )
            )

        with tf.variable_scope('dense'):
            self.logger.debug('Creating dense layers')
            lstm_concat = tf.concat(lstm_outputs, axis=1)

            dense_1 = tf.layers.dense(
                lstm_concat, units=CONST['LSTM_SIZE'], activation=tf.nn.relu)

            self.logits = tf.layers.dense(dense_1, 2)

        self.labels = tf.placeholder(tf.int64, shape=[self.batch_size], name='Y')
        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels))
        self.probabilities = tf.nn.softmax(self.logits)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.loss_summary = tf.summary.scalar('sigmoid_mean_loss', self.loss)

        self.summaries = tf.summary.merge_all()

        self.logger.debug('==============Model built==============')

    def sentence_lstm(self, embeddings, reuse=False):
        self.logger.debug('Creating LSTM layer')
        with tf.variable_scope('sentence_LSTM', reuse=reuse):
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                self.forward_cell,
                self.backward_cell,
                embeddings,
                dtype=tf.float32)

        rnn_outputs = tf.concat([rnn_outputs[0], rnn_outputs[1]], axis=1)

        attention_size = 50
        with tf.variable_scope('sentence_attention', reuse=reuse):
            W_attention = tf.get_variable(
                'W_attention', shape=[CONST['LSTM_SIZE'], attention_size], initializer=xavier_initializer(), )
            b_attention = tf.get_variable(
                'b_attention', shape=[attention_size], initializer=xavier_initializer())
            u_attention = tf.get_variable(
                'u_attention', shape=[attention_size], initializer=xavier_initializer())
            v = tf.tanh(tf.tensordot(rnn_outputs, W_attention, axes=1) + b_attention)

            vu = tf.tensordot(v, u_attention, axes=1, name='vu')
            alphas = tf.nn.softmax(vu, name='alphas')

            output = tf.reduce_sum(rnn_outputs * tf.expand_dims(alphas, -1), 1)

        return output
