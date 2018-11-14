import tensorflow as tf
from .action import Action


class Trainer(Action):
    def __init__(self, sess, model, reader, args, logger):
        super().__init__(sess, model, reader, args, logger)

        self.saver = tf.train.Saver(max_to_keep=1)
        #self.train_step = self.get_train_step()
        #self.best_validation = 0.0
        #self.load_model()
        self.logger.log('============Trainer initialized============')

    def train(self):
        self.logger.log('Starting training...')
        """
        for cur_epoch in range(self.args.epochs):
            global_step = self.sess.run(self.model.global_step_tensor)
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.validate(global_step)
        """
        self.logger.log('Training finished.')

    def train_epoch(self, epoch):
        next_batch = self.reader.get_iterator()
        while True:
            try:
                sentences, labels, indices = self.sess.run(next_batch)
                loss, accuracy, global_step, summaries, _ = self.sess.run(
                    [self.model.loss, self.model.accuracy, self.model.global_step_tensor,
                        self.model.summaries, self.train_step],
                    feed_dict={
                        self.model.sentences: sentences,
                        self.model.labels: labels,
                        self.model.training: True,
                        self.model.encodings: [self.reader.training_encodings[index]
                                               for index in indices]
                    })

                self.debugger.debug(
                    f'Epoch: {epoch}. Global step {global_step}. Loss: {loss}. Accuracy: {accuracy}')
                if global_step % 100 == 0:
                    self.logger.summarize(summaries, global_step)
                    self.validate(global_step)

            except tf.errors.OutOfRangeError:
                break

    # save function that saves the checkpoint in the path defined in the config file
    def save_model(self, global_step):
        self.debugger.debug(f"Saving model. Global step: {global_step}")
        self.saver.save(self.sess, self.args.checkpoint_dir, self.model.global_step_tensor)
        self.debugger.debug(f"Model {global_step} saved")

    def load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_dir)
        if latest_checkpoint:
            self.debugger.debug(f"Loading model from {latest_checkpoint}")
            self.sess.run(tf.tables_initializer())
            self.saver.restore(self.sess, latest_checkpoint)
            self.debugger.debug("Model loaded")
        else:
            self.debugger.debug('Initializing new model')
            self.init = [
                tf.global_variables_initializer(),
                tf.tables_initializer()]
            self.sess.run(self.init)

            self.debugger.debug('New model initialized')

    def validate(self, global_step):
        next_batch = self.reader.get_iterator(dataset='validation')
        total_stories = 0
        total_correct = 0
        while True:
            try:
                sentences, labels, indices = self.sess.run(next_batch)
                probabilities = self.sess.run(
                    [self.model.probabilities],
                    feed_dict={
                        self.model.sentences: sentences,
                        self.model.encodings: [self.reader.validation_encodings[index]
                                               for index in indices]
                    })
                for i in range(0, len(probabilities[0]), 2):
                    assert labels[i] + labels[i + 1] == 1
                    true_label = 0 if labels[i] else 1
                    prob1 = probabilities[0][i][1]
                    prob2 = probabilities[0][i + 1][1]
                    label = 0 if prob1 > prob2 else 1
                    if true_label == label:
                        total_correct += 1
                    total_stories += 1
            except tf.errors.OutOfRangeError:
                break
        accuracy = total_correct / total_stories
        if accuracy > self.best_validation:
            self.best_validation = accuracy
            self.save_model(global_step)
        self.debugger.debug(f'Validation accuracy: {accuracy}')
        summary = tf.Summary(value=[tf.Summary.Value(tag='validation_accuracy',
                                                     simple_value=accuracy)])
        self.logger.summarize(summary, global_step)

    def get_train_step(self):
        optimizer = tf.train.AdamOptimizer()
        params = tf.trainable_variables()
        gradients = tf.gradients(self.model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        return optimizer.apply_gradients(
            zip(clipped_gradients, params), global_step=self.model.global_step_tensor)
