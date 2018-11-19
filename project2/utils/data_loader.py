import os
import re
import sys
import tensorflow as tf

from .datapreparation import visualize_piano_roll, load_all_dataset

import pandas as pd
import pickle
from utils.constants import CONST
import tensorflow as tf
import numpy as np


class Reader():
    def __init__(self, args, sess, logger):
        self.args = args
        self.sess = sess
        self.logger = logger

        self.logger.log('============Preparing datasets============')

        piano_roll = load_all_dataset('datasets/training/piano_roll_fs1_subset/')
        print(piano_roll)
        piano_roll = piano_roll[0]
        print(piano_roll.shape)
        piano_roll = tf.data.Dataset.from_tensor_slices(piano_roll)
        self.dataset = tf.data.Dataset.batch(piano_roll, self.args.batch_size, drop_remainder=True)

        print(self.dataset)
        self.logger.log('============Datasets prepared============')

    def process_dataset(self, path, encodings_path, shuffle=False, is_validation=False, is_prediction=False):
        dataset, labels = self._load_data(path, is_prediction)
        print(dataset)
        index = tf.data.Dataset.from_tensor_slices([i for i in range(len(dataset))])
        print(f"index: {index}")

        encodings = np.array(self.get_encodings(dataset, encodings_path))
        encodings = np.reshape(encodings, (-1, 5, CONST['SKIP_SIZE']))

        dataset = self._preprocess(dataset, validation=is_validation)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        # dataset_lengths = tf.data.Dataset.from_tensor_slices(dataset_lengths)
        if not is_prediction:
            labels = tf.data.Dataset.from_tensor_slices(labels)
            # encodings = tf.data.Dataset.from_tensor_slices(encodings)

            dataset = tf.data.Dataset.zip((dataset, labels, index))
        if not is_validation:
            dataset = dataset.shuffle(buffer_size=150)
        if is_prediction:
            # handle last batch here
            print('we need to handle batch size')
            print(dataset)

            dataset = tf.data.Dataset.zip((dataset, index))

            dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(self.args.batch_size))
        else:
            dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(self.args.batch_size))
        return dataset, labels, encodings

    def get_encodings(self, dataset, filepath):
        if os.path.isfile(filepath):
            self.debugger.debug(f'Found skip thought embeddings at {filepath}.')
            return pickle.load(open(filepath, 'rb'))

        np_dataset = np.array(dataset).reshape((-1))
        encodings = self.encoder.encode(np_dataset)

        encodings = np.array(encodings).reshape((-1, 5, 2400))
        self.debugger.debug(f'Saved skip thought file to {filepath}')
        pickle.dump(encodings, open(filepath, 'wb'), protocol=4)
        return encodings

    def get_iterator(self, dataset='training'):
        if dataset == 'training':
            return self.training_dataset.make_one_shot_iterator().get_next()
        elif dataset == 'validation':
            return self.validation_dataset.make_one_shot_iterator().get_next()
        else:
            return self.test_dataset.make_one_shot_iterator().get_next()

    def _parse_sentence(self, dataset):
        def parse_sentence(sentence):
            sentence = re.sub('[^0-9a-zA-Z ]+', '', sentence)
            sentence = sentence.lower()
            sentence = sentence.strip()
            return sentence
        return self._transform_data(dataset, parse_sentence)

    def _preprocess(self, dataset, validation=False):
        """
        Applies all transformations to preprocess data inplace
        """
        self.debugger.debug('Starting data preprocessing...')

        dataset = self._parse_sentence(dataset)

        if not validation:
            self.word_to_index = self._get_vocabulary(dataset)
            self.index_to_word = {}
            for word in self.word_to_index.keys():
                self.index_to_word[self.word_to_index[word]] = word
        dataset = self._replace_uncommon_words(dataset, self.word_to_index)

        # dataset_lengths, max_length = self._get_sentence_lengths(dataset)
        dataset = self._pad_sentences(dataset)
        dataset = self._map_to_indices(dataset)

        self.debugger.debug('Data preprocessing finished!')
        return dataset

    def _get_vocabulary(self, corpus):
        """
        Creates a dictionary with the words and their counts, only the most frequent ones are preserved
        :param corpus: the dataset used to create the vocabulary
        :return: the dictionary with words and counts
        """
        self.debugger.debug('Creating word_to_index from training corpus...')

        if os.path.isfile(CONST['VOCAB_FILEPATH']):
            self.debugger.debug(f'Found word_to_index vocabulary at {CONST["VOCAB_FILEPATH"]}.')
            return pickle.load(open(CONST['VOCAB_FILEPATH'], 'rb'))

        elif self.args.action != 'train':
            raise ImportError(
                'No vocabulary file found. Please run training first to generate a vocab')

        self.debugger.debug(f'No vocabulary file found. Creating vocabulary from training data.')
        corpus = np.reshape(corpus, -1)
        corpus = ' '.join(corpus)

        word_count = {}
        for word in corpus.split(' '):
            if word_count.get(word):
                word_count[word] += 1
            else:
                word_count[word] = 1

        vocab_list = [(key, value)
                      for key, value in word_count.items() if value >= CONST['MIN_FREQUENCY']]
        vocab_list.sort(key=lambda tuple: tuple[1], reverse=True)
        # Subtract 2 to make room for <unk> and <pad>
        vocab_list = vocab_list[: self.args.vocab_size - 2]
        vocab_list.append((CONST['UNKNOWN_WORD'], CONST['MIN_FREQUENCY']))
        vocab_list.append((CONST['PAD'], CONST['MIN_FREQUENCY']))

        word_to_index = {}
        for index, key_value in enumerate(vocab_list):
            word_to_index[key_value[0]] = index

        self.debugger.debug(f'Saved vocabulary file to {CONST["VOCAB_FILEPATH"]}')
        pickle.dump(word_to_index, open(CONST['VOCAB_FILEPATH'], 'wb'))
        return word_to_index

    # def _pad_special_chars(self, dataset):
    #     # add spaces around the special characters so they don't affect
    #     # regular words when generating vocabulary
    #     def replace_chars(s):
    #         special_chars = ['.', '!', '?']

    #         for char in special_chars:
    #             s = s.replace(char, ' ' + char + ' ')
    #         return s

    #     padded_dataset = self._transform_data(dataset, replace_chars)

    #     # There might be two spaces in a row now, or a space at the end of a sentence.
    #     # This will create an empty string element when splitting on spaces. We remove them:
    #     def remove_extra_spaces(s):
    #         return re.sub(r' +', ' ', s).strip()

    #     return self._transform_data(padded_dataset, remove_extra_spaces)

    def _load_data(self, filepath, is_test=False):
        self.debugger.debug('Loading data from: ' + filepath)
        dataset, labels = self._load_csv(filepath, is_test)
        self.debugger.debug('Data loaded.')
        return dataset, labels

    def _load_csv(self, filepath, is_test=False):
        if is_test:
            print('inside is_test')
            X = np.array(pd.read_csv(filepath).values.tolist())
            print(f"X in _load__csv: {X.shape}")
            data = X[:, :5]
            print(f"data in _load__csv: {data.shape}")
            labels = None
        else:
            X = np.array(pd.read_csv(filepath, usecols=CONST['COLS']).values.tolist())
            data = X[:, :5]
            labels = X[:, 5].astype(int)
        return data, labels

    # def _replace_entities(self, dataset, filepath):
    #     """
    #     Replace the words with an ER.
    #     :return: Dataframe with words replaced by their corresponding ER
    #     """

    #     if os.path.isfile(filepath):
    #         self.debugger.debug(f'Found SpaCy preprocessed data file at {filepath}')
    #         return np.load(filepath)  # we already have the preprocessed data

    #     self.debugger.debug(f'No preprocessed data file found: {filepath}')

    #     def replace_named_entities(s):
    #         doc = self.nlp(str(s))
    #         new_str = []
    #         for word in doc:
    #             if word.ent_type_ in CONST['NER_TAGS']:
    #                 new_str.append('<' + word.ent_type_ + '>')
    #             else:
    #                 new_str.append(word.text)
    #         return ' '.join(new_str)

    #     # obtain all NER tags
    #     data = self._transform_data(dataset, replace_named_entities)

    #     # preprocessing is expensive, let's save it in a file
    #     np.save(filepath, data)
    #     return data

    def _replace_uncommon_words(self, data, vocabulary):
        def replace_word_with_unk(s):
            split_sentence = s.split(' ')
            transformed_sentence = []
            for word in split_sentence:
                if vocabulary.get(word) is not None:
                    transformed_sentence.append(word)
                else:
                    transformed_sentence.append(CONST['UNKNOWN_WORD'])
            return ' '.join(transformed_sentence)
        return self._transform_data(data, replace_word_with_unk)

    def _to_lower_case(self, data):
        return self._transform_data(data, str.lower)

    def _transform_data(self, data, fn):
        vec_fn = np.vectorize(fn)
        return vec_fn(data)

    def _get_sentence_lengths(self, dataset):
        dataset_lengths = []
        max_length = 0
        for story in dataset:
            lengths = [*map(lambda x: len(x.split(' ')), [sentence for sentence in story])]
            max_length = max(max_length, *lengths)
            dataset_lengths.append(lengths)
        return np.array(dataset_lengths, dtype=np.int32), max_length

    def _pad_sentences(self, dataset):
        padded_stories = []
        for story in dataset:
            padded_sentences = []
            for sentence in story:
                words = sentence.split()[: self.args.sentence_length]
                padded_words = [*words, *[CONST['PAD']
                                          for _ in range(self.args.sentence_length - len(words))]]
                padded_sentences.append(padded_words)
            padded_stories.append(padded_sentences)
        return np.array(padded_stories)

    def _map_to_indices(self, dataset):
        return self._transform_data(dataset, lambda w: self.word_to_index[w])
