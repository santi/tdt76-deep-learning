import tensorflow as tf
import numpy as np
import os
from .action import Action


class Predictor(Action):
    def __init__(self, sess, model, reader, args, logger):
        super().__init__(sess, model, reader, args, logger)
        
        self.load_model()

    def predict(self):
        next_batch = self.reader.get_iterator(dataset='predict')
        while True:
            try:
                sentences, indices = self.sess.run(next_batch)
                print(sentences)
                print(indices)
                probabilities = self.sess.run(
                    [self.model.probabilities],
                    feed_dict={
                        self.model.sentences: sentences,
                        self.model.training: False,
                        self.model.encodings: [self.reader.test_encodings[index]
                                               for index in indices]
                    })
                probabilities = probabilities[0]  # np.squeeze(probabilities, axis=1)
                print(sentences.shape)
                print(probabilities)
                print(probabilities.shape)
                assert len(probabilities) == 2
                prob1 = probabilities[0][1]
                prob2 = probabilities[1][1]
                label = 1 if prob1 > prob2 else 2

                with open('data/prediction.csv', 'a') as f:
                    np.savetxt(f, [label], delimiter=',', fmt='%i')

            except tf.errors.OutOfRangeError:
                break

    def load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.args.checkpoint_dir, self.args.model_name))
        if latest_checkpoint:
            self.logger.load_checkpoint(latest_checkpoint)
        else:
            raise NotImplementedError()
