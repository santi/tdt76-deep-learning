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

        piano_rolls = load_all_dataset('data/training/piano_roll_fs1/')

        #test_roll = piano_rolls[0].T
        #visualize_piano_roll(test_roll)
        #print(test_roll[1])

        features_dataset = []
        labels_dataset = []
        for roll in piano_rolls:
            features = roll.T
            labels = features[1:]
            # pad labels with '1' to symbolize end of song
            labels = np.append(labels, [np.ones_like(labels[0])], axis=0)
            #print(f'song shape: {features.shape}')
            
            num_samples = len(features) // CONST['SEQUENCE_LENGTH']
            #print(f'This will generate {num_samples} samples')
            end_index = num_samples * CONST['SEQUENCE_LENGTH']

            features = features[:end_index]
            labels = labels[:end_index]
            for i in range(num_samples):
                features_dataset.append(features[i * CONST['SEQUENCE_LENGTH'] : i * CONST['SEQUENCE_LENGTH'] + CONST['SEQUENCE_LENGTH']])
                labels_dataset.append(labels[i * CONST['SEQUENCE_LENGTH'] : i * CONST['SEQUENCE_LENGTH'] + CONST['SEQUENCE_LENGTH']])

        self.features_array = self._batch(np.array(features_dataset))
        self.labels_array = self._batch(np.array(labels_dataset))

        self.training_data = (self.features_array, self.labels_array)

        self.logger.log('============Datasets prepared============')


    def get_iterator(self, dataset='training'):
        if dataset == 'training':
            return self.training_data
        elif dataset == 'validation':
            return self.validation_dataset.make_one_shot_iterator()
        #else:
        #    return self.test_dataset.make_one_shot_iterator().get_next()


    def _batch(self, dataset, drop_remainder=True):
        num_batches = len(dataset) // self.args.batch_size
        batches = []
        for i in range(num_batches):
            batches.append(dataset[i * self.args.batch_size : i * self.args.batch_size + self.args.batch_size])
        return np.array(batches)


    def _load_data(self, filepath, is_test=False):
        self.debugger.debug('Loading data from: ' + filepath)
        dataset, labels = self._load_csv(filepath, is_test)
        self.debugger.debug('Data loaded.')
        return dataset, labels


    def _transform_data(self, data, fn):
        vec_fn = np.vectorize(fn)
        return vec_fn(data)
