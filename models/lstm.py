import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from .model import Model
from utils.constants import CONST




class VanillaLSTM(Model):
    def __init__(self, args, logger):
        super().__init__(args, logger)

        self.batch_size = self.args.batch_size
        self.sequence_length = CONST['SEQUENCE_LENGTH']

        if self.args.action == 'predict':
            self.batch_size = 1
            self.sequence_length = 1

        self.build_model()

    def build_model(self):
        self.logger.log('==============Building model==============')

        self.X = tf.placeholder(tf.float32, [
            self.batch_size,
            self.sequence_length,
            CONST['NOTE_LENGTH']], name='X')

        if self.args.action == 'train_composer':
            self.initial_hidden_state = tf.get_variable(
                'initial_hidden_state', shape=[self.batch_size, CONST['LSTM_SIZE']], dtype=tf.float32, trainable=True)
            self.initial_cell_state = tf.get_variable(
                'initial_cell_state', shape=[self.batch_size, CONST['LSTM_SIZE']], dtype=tf.float32, trainable=True)
        else:
            self.initial_hidden_state = tf.placeholder(
                tf.float32, shape=[self.batch_size, CONST['LSTM_SIZE']], name='initial_hidden_state')
            self.initial_cell_state = tf.placeholder(
                tf.float32, shape=[self.batch_size, CONST['LSTM_SIZE']], name='initial_cell_state')

        with tf.variable_scope('lstm_cell'):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=CONST['LSTM_SIZE'])
            initial_state = (self.initial_cell_state, self.initial_hidden_state)

        with tf.variable_scope('LSTM') as scope:
            outputs = []
            for i in range(self.sequence_length):
                if i == 0:
                    output, state = cell(inputs=self.X[:, i, :], state=initial_state)
                else:
                    scope.reuse_variables()
                    output, state = cell(inputs=self.X[:, i, :], state=state)
                outputs.append(output)
            self.output_state = state
            outputs = tf.stack(outputs, axis=1)
            print(f"LSTM outputs: {outputs.shape}")


        self.logits = tf.layers.dense(
            inputs=outputs,
            units=CONST['NOTE_LENGTH'],
            name='dense')

        print(f"logits: {self.logits.shape}")
        self.outputs = tf.nn.sigmoid(self.logits)
        
        
        self.Y = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.sequence_length, CONST['NOTE_LENGTH']),
            name='Y')
        
        print(f"Y: {self.Y.shape}")
        
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.Y, logits=self.logits))
            self.loss_summary = tf.summary.scalar('sigmoid_mean_loss', self.loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer()
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.train_step = self.optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step_tensor)
        
        self.summaries = tf.summary.merge_all()
        self.logger.log('==============Model built==============')
