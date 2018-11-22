import tensorflow as tf
import numpy as np

class Action:
    def __init__(self, sess, model, reader, args, logger):
        self.sess = sess
        self.model = model
        self.reader = reader
        self.args = args
        self.logger = logger

    def load_model(self):
        raise NotImplementedError('Action subclass must override load_model')