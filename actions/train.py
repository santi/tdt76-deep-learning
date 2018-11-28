import tensorflow as tf
from .action import Action
import numpy as np
import time
import sys
import os
from utils.constants import CONST


class Trainer(Action):
    def __init__(self, sess, model, reader, args, logger):
        super().__init__(sess, model, reader, args, logger)

        self.load_model()
        self.logger.log('============Trainer initialized============')


    def train(self):
        self.logger.log('============Starting training============')
        for cur_epoch in range(self.args.epochs):
            self.train_epoch(cur_epoch)
            self.validate()
            self.sess.run(self.model.increment_cur_epoch_tensor)
        
        self.logger.log('============Training finished============')


    def train_epoch(self, epoch):
        batch_features, batch_labels = self.reader.get_iterator()
        initial_state = np.zeros((self.args.batch_size, CONST['LSTM_SIZE']))
        for i in range(len(batch_features)):
            self.sess.run(
                self.model.train_step,
                feed_dict={
                    self.model.X: batch_features[i],
                    self.model.Y: batch_labels[i],
                    self.model.initial_hidden_state: initial_state,
                    self.model.initial_cell_state: initial_state,
                })


    # save function that saves the checkpoint in the path defined in the config file
    def save_model(self):
        global_step = self.sess.run(self.model.global_step_tensor)
        self.logger.save_checkpoint(global_step)


    def load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.args.checkpoint_dir, self.args.model_name))
        if latest_checkpoint:
            self.logger.load_checkpoint(latest_checkpoint)
        else:
            self.logger.log('Initializing new model')
            self.sess.run(tf.global_variables_initializer())
            self.logger.log('New model initialized')
        self.best_validation = self.sess.run(self.model.validation_score)
        self.logger.log(f"Best validation score: {self.best_validation}")


    def print_note(self, logits):
        func = lambda x: '%.2f' % float(x)
        print(' '.join(map(func, logits)))


    def validate(self):
        total_loss = 0.
        batch_features, batch_labels = self.reader.get_iterator('validation')

        initial_state = np.zeros((self.args.batch_size, CONST['LSTM_SIZE']))
        for i in range(len(batch_features)):
            total_loss += self.sess.run(
                self.model.loss,
                feed_dict={
                    self.model.X: batch_features[i],
                    self.model.Y: batch_labels[i],
                    self.model.initial_hidden_state: initial_state,
                    self.model.initial_cell_state: initial_state,
                })

        if total_loss < self.best_validation:
            self.best_validation = total_loss
            self.sess.run(self.model.assign_validation_score, feed_dict={
                self.model.set_validation_score: total_loss,
            })
            self.save_model()
        


        global_step, epoch, summaries, outputs = self.sess.run(
                [
                    self.model.global_step_tensor,
                    self.model.cur_epoch_tensor,
                    self.model.summaries,
                    self.model.outputs,
                ],
                feed_dict={
                    self.model.X: batch_features[-1],
                    self.model.Y: batch_labels[-1],
                    self.model.initial_hidden_state: initial_state,
                    self.model.initial_cell_state: initial_state,
                })
        self.print_note(outputs[0][0][50:90])
        self.print_note(batch_labels[-1][0][0][50:90])

        self.logger.log(f'Finished epoch {epoch}. Global step {global_step}. Total validation loss: {total_loss}')
        total_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='total_validation_loss',
                                                     simple_value=total_loss)])
        self.logger.summarize(total_loss_summary, global_step)
        self.logger.summarize(summaries, global_step)
