from base.base_model import BaseModel
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.layers import xavier_initializer

from utils.constants import CONST
from .conv_embedding import ConvolutionalEmbedding as conv_emb

"""
    The idea is that the correct phrase is correct because it has elements that were mentioned in the four sentences.
    This means that it matches the best the concept of the 4 previous sentences. If a new element is introduced
    then this causes some surprise and this is more unlikely to happen.
    
    We create 4 recursive neural networks. One for each of the sentences. We use recursive neural networks
    because we will input the phrases according to the parse grammar because it makes more sense to follow the
    structure of the phrase. From this the idea is to extract an embedding, in a similar way to how they do this
    for the sentiment analysis where they also use this tree-like processing.
    Some papers have shown that for sentence embeddings this recursive idea works quite well. Please refer to
    [citation needed].
    Once we have each of these 4 embeddings we use an RNN to process the sentences sequentially. This makes sense
    because the sentences have this natural order. We don't need a recursive network. From this RNN we will obtain
    an embedding the same size as the embedding produced by the recursive neural networks.
    We finally compare the embedding from the 4 sentences against the embedding from the candidates. The candidate
    that is the closest to the embedding produced by the RNN is the correct one.
    
    A modification could be stopping at the recursive neural networks and compare directly against the candidates'
    embeddings. The one that has the lowest sum of errors is the correct one. I think this might not peform as
    good as the original idea.
    
    For the vanilla approach we will just use 4 RNNs for each sentence and compare against the candidate.
    An improved vanilla approach could be using 4 RNNs and then an RNN on top of that one and compare
    against the candidates.
    Replace the RNNs with recursive nets to obtain the final approach. Reading the tree is the real challenge here.
"""


class RecEmbeddingRNN(BaseModel):
    def __init__(self, args, debugger):
        super(RecEmbeddingRNN, self).__init__(args, debugger)
        self.sentence_embeddings = [a.lower() for a in args.sentence_embeddings]
        self.build_model()

    def build_model(self):
        # TODO: Change method to allow testing of model, not just training

        self.debugger.debug('==============Building model==============')
        # create word embeddings for the words in the sentences.
        with tf.variable_scope('word_embeddings'):
            self.debugger.debug('Creating word embedding layer')
            
            self.W_embedding = self.create_embeddings()

            self.sentences = tf.placeholder(tf.int32, [self.batch_size, 5, None], name='sentences')
            self.sentence_lengths = tf.placeholder(tf.int32, [self.batch_size, 5], name='sentence_lengths')
            self.original_sentences = tf.placeholder(tf.string, [self.batch_size, 5], name='original_sentences')

            self.embeddings = []
            for sentence_index in range(self.sentences.shape[1]):
                self.embeddings.append(
                    (tf.nn.embedding_lookup(
                        self.W_embedding,
                        self.sentences[:, sentence_index],
                        name=f'embedding_{sentence_index}'),
                    self.sentence_lengths[:, sentence_index],
                    self.original_sentences[:, sentence_index]))

        # create rnns with shared weights
        
        with tf.variable_scope('sentence_embeddings'):
            s_emb_sets = []
            all_s_emb = 'all' in self.sentence_embeddings #shorthand

            if all_s_emb or 'lstm' in self.sentence_embeddings:
                s_emb_sets.append(self.s_emb_lstm())
                
            if all_s_emb or 'google' in self.sentence_embeddings:
                s_emb_sets.append(self.s_emb_google())
                
            if all_s_emb or 'conv' in self.sentence_embeddings:
                s_emb_sets.append(self.s_emb_conv())

        with tf.variable_scope('Sentence_level_embedding_concatenation'):
            # Make a list where the ith entry is the concatenated embeddings 
            # for the ith sentence
            embedded_story = [
                tf.concat([emb[i] for emb in s_emb_sets], axis=1) 
                for i in range(5)]

        with tf.variable_scope('dense'):
            self.debugger.debug('Creating dense layers')
            concatenated_emb_dims = embedded_story[0].shape[1]

            first_part = tf.concat(embedded_story[:4], axis=1)
            story_embedding = tf.layers.dense(
                first_part, units=concatenated_emb_dims, activation=tf.nn.relu)
            last_part = embedded_story[4]
            self.logits = tf.layers.dense(
                tf.concat([story_embedding, last_part], axis=1), 1, activation=None)

            self.probabilities = tf.nn.sigmoid(self.logits, name='output')

        self.labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1), name='Y')
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels, name='loss'))
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.cast(self.labels, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.loss_summary = tf.summary.scalar('sigmoid_mean_loss', self.loss)

        self.summaries = tf.summary.merge_all()

        self.debugger.debug('==============Model built==============')


    def s_emb_lstm(self):
        with tf.variable_scope('LSTMs'):
            self.debugger.debug('Creating LSTM layer')
            with tf.variable_scope('LSTM_cell'):
                lstm_cell = tf.nn.rnn_cell.LSTMCell(CONST['LSTM_SIZE'])
                initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

                # TODO: remove. Used to make Tensorboard look a lot prettier, and makes the inspection easier.
                lstm_cell(tf.placeholder(tf.float32, shape=(
                    self.batch_size, self.embedding_size)), initial_state)

            lstm_outputs = []
            for sentence, length, _ in self.embeddings:
                dynamic_rnn_outputs, final_state = tf.nn.dynamic_rnn(
                    lstm_cell,
                    sentence,
                    initial_state=initial_state,
                    sequence_length=length)
                # Only append the final output from the RNN, before padding begins
                lstm_outputs.append(final_state.h) 
    
            return lstm_outputs

    
    def s_emb_google(self):
        with tf.variable_scope('Google_sentence_embedding'):
            sentence_embedder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/1")
            
            google_sentence_encodings = []
            for _, _, original_sentence in self.embeddings:
                google_sentence_encodings.append(sentence_embedder(original_sentence))
            return google_sentence_encodings


    def s_emb_conv(self):
        with tf.variable_scope('ConvolutionalEmbedding'):
            s_embeddings = []
            for sentence, length, _ in self.embeddings:
                embedding = conv_emb.get_convolutional_embedding(sentence)
                s_embeddings.append(embedding)
            return s_embeddings



    def create_embeddings(self):
        return tf.get_variable(
            'W_embedding',
            [self.vocab_size, self.embedding_size],
            initializer=xavier_initializer())
