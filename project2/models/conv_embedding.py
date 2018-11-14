import tensorflow as tf
from utils.constants import CONST


class ConvolutionalEmbedding(object):
    @staticmethod
    def get_convolutional_embedding(sentences):
        """
        Creates an embedding of predefined-size for a column of sentences
        :param sentences: sentences of shape [batch_size, max_words, embedding_size]. It should
        be padded to max_words
        :param lengths: integer that represents the maximum number of words in 'sentences'.
        :return:[batch_size, len(CONST['CONV_FILTER_SIZES'])*CONST['CONV_OUT_SIZE']
        """
        with tf.variable_scope('Convolutional_Layer', reuse=tf.AUTO_REUSE):
            feature_maps = []
            for height in CONST['CONV_FILTER_SIZES']:
                conv = tf.layers.conv1d(sentences, CONST['CONV_OUT_SIZE'], height, 1, padding='valid',
                                        name=f'c_{height}',activation=tf.nn.relu)
                # simulate pool, since we can't use dynamic size for pooling
                pool = tf.reduce_max(conv, axis=1, name=f'p_{height}')
                if CONST['CONV_DROPOUT'] > 0:
                    pool = tf.nn.dropout(pool, CONST['CONV_DROPOUT'], name=f'd_{height}')
                feature_maps.append(pool)
            return tf.concat(feature_maps, axis=1, name='feature_maps')
