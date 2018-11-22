import os
import re
import sys
import tensorflow as tf

from .datapreparation import visualize_piano_roll, load_all_dataset

import pandas as pd
from utils.constants import CONST
import tensorflow as tf
import numpy as np


class Reader():
    def __init__(self, args, sess, logger):
        self.args = args
        self.sess = sess
        self.logger = logger

        self.logger.log('============Preparing datasets============')

        piano_rolls = self._load_data(self.args.training_data)

        #test_roll = piano_rolls[0].T
        #visualize_piano_roll(test_roll)
        #print(test_roll[1])

        self.features_dataset = []
        self.labels_dataset = []
        for roll in piano_rolls:
            features = roll.T
            labels = features[1:]
            labels = np.append(labels, [np.ones_like(labels[0])], axis=0)

            assert features.shape == labels.shape
            
            for i in range(len(features) - CONST['SEQUENCE_LENGTH']):
                self.features_dataset.append(features[i: i + CONST['SEQUENCE_LENGTH']])
                self.labels_dataset.append(labels[i: i + CONST['SEQUENCE_LENGTH']])


        self.features_dataset = np.array(self.features_dataset)
        self.labels_dataset = np.array(self.labels_dataset)

        self.logger.log('============Datasets prepared============')


    def get_iterator(self, dataset='training'):
        if dataset == 'training':
            return self._shuffle_and_batch(self.features_dataset, self.labels_dataset)
        else:
            raise NotImplementedError('Must get training data')
        #elif dataset == 'validation':
        #    return self.validation_dataset.make_one_shot_iterator()
        #else:
        #    return self.test_dataset.make_one_shot_iterator().get_next()


    def _shuffle_and_batch(self, a, b):
        self._shuffle_in_unison(a, b)

        features_array = self._batch(self.features_dataset)
        labels_array = self._batch(self.labels_dataset)

        return (features_array, labels_array)


    def _shuffle_in_unison(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)


    def _batch(self, dataset, drop_remainder=True):
        num_batches = len(dataset) // self.args.batch_size
        batches = []
        for i in range(num_batches):
            batches.append(dataset[i * self.args.batch_size : i * self.args.batch_size + self.args.batch_size])
        return np.array(batches)


    def _load_data(self, filepath):
        self.logger.log('Loading data from: ' + filepath)
        data = load_all_dataset(filepath)
        self.logger.log('Data loaded.')
        return data

