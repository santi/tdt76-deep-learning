import tensorflow as tf
import numpy as np
import os
from .action import Action
from utils.constants import CONST
from utils.datapreparation import visualize_piano_roll, load_all_dataset


class Predictor(Action):
    def __init__(self, sess, model, reader, args, logger):
        super().__init__(sess, model, reader, args, logger)
        
        self.load_model()

    def predict(self):
        self.logger.log('Start generating songs...')
        features = self.reader.get_iterator(dataset='prediction')
        print(f"predict feature shape: {features.shape}")
        predicted_songs = []
        for roll in features:
            print(f"roll shape: {roll.shape}")
            counter = 0
            notes = []
            cell_state = np.zeros((1, CONST['LSTM_SIZE']))
            hidden_state = np.zeros((1, CONST['LSTM_SIZE']))
            state = (cell_state, hidden_state)

            first = True
            for note in roll:
                print(f"note shape: {note.shape}")
                reshaped_note = np.reshape(note, (1, 1, CONST['NOTE_LENGTH']))
                print(f"note reshaped: {reshaped_note.shape}")
                if first:
                    first = False
                    logits, state = self.sess.run(
                        [self.model.outputs,
                        self.model.output_state],
                        feed_dict={
                            self.model.X: reshaped_note,
                            self.model.initial_hidden_state: state[0],
                            self.model.initial_cell_state: state[1],
                        })
                    print(f"note output shape: {note.shape}")
                    notes.append(note)
                else:
                    logits, state = self.sess.run(
                        [self.model.outputs,
                        self.model.output_state],
                        feed_dict={
                            self.model.X: reshaped_note,
                            self.model.initial_cell_state: state.c,
                            self.model.initial_hidden_state: state.h,
                        })
                    print(note.shape)
                    notes.append(note)

            
            while counter < 20: # TODO: change to 100
                note = np.reshape(notes[-1], (1, 1, CONST['NOTE_LENGTH']))
                logits, state = self.sess.run(
                    [self.model.outputs,
                    self.model.output_state],
                    feed_dict={
                        self.model.X: note,
                        self.model.initial_cell_state: state.c,
                        self.model.initial_hidden_state: state.h,
                    })
                notes.append(np.reshape(logits, (128,)))
                counter += 1
            
            notes = np.array(notes)
            print(f'output notes shape: {notes.shape}')
            for note in notes:
                print(note[50:90])
            visualize_piano_roll(notes)

    def load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.args.checkpoint_dir, self.args.model_name))
        if latest_checkpoint:
            self.logger.load_checkpoint(latest_checkpoint)
        else:
            raise NotImplementedError()
