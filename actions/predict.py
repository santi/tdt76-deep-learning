import tensorflow as tf
import numpy as np
import os
from .action import Action
from utils.constants import CONST
from utils.datapreparation import visualize_piano_roll, load_all_dataset, piano_roll_to_mid_file


class Predictor(Action):
    def __init__(self, sess, model, reader, args, logger):
        super().__init__(sess, model, reader, args, logger)
        
        self.load_model()

    def predict(self):
        self.logger.log('Start generating songs...')
        features = self.reader.get_iterator(dataset='prediction')
        predicted_songs = []
        for roll in features:
            notes = []
            state = self._get_hidden_state()

            for note in roll:
                reshaped_note = np.reshape(note, (1, 1, CONST['NOTE_LENGTH']))
                state, _ = self.sess.run([
                    self.model.output_state,
                    self.model.outputs,
                    ],
                    feed_dict={
                        self.model.X: reshaped_note,
                        self.model.initial_cell_state: state.c,
                        self.model.initial_hidden_state: state.h,
                    })
                notes.append(note) # Append the original note

            end_signal = np.ones((128,))
            while len(notes) < 100:
                note = np.reshape(notes[-1], (1, 1, CONST['NOTE_LENGTH']))
                outputs, state = self.sess.run(
                    [
                        self.model.outputs,
                        self.model.output_state
                    ],
                    feed_dict={
                        self.model.X: note,
                        self.model.initial_cell_state: state.c,
                        self.model.initial_hidden_state: state.h,
                    })

                outputs = np.reshape(outputs, (128,))
                predicted_note = outputs > CONST['NOTE_THRESHOLD']
                outputs = np.zeros_like(outputs)
                outputs[predicted_note] = 1

                if np.array_equal(outputs, end_signal):
                    break
                
                notes.append(outputs)

            predicted_songs.append(np.array(notes) * 100)
        self.logger.log('Finished generating songs')

        self.logger.log('Saving songs...')
        for i, song in enumerate(predicted_songs):
            self.logger.log(f'Saving song {i} shape: {song.shape}')
            visualize_piano_roll(song, save=True, filename=f'song-{i}')
            piano_roll_to_mid_file(song, f"{os.path.join(self.args.output_dir, self.args.model_name)}/song-{i}.mid", fs=5)
        self.logger.log('Finished saving songs')

    def load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.args.checkpoint_dir, self.args.model_name))
        if latest_checkpoint:
            self.logger.load_checkpoint(latest_checkpoint)
        else:
            raise NotImplementedError()

    def _get_hidden_state(self):
        cell_state = None
        hidden_state = None
        if self.args.composer:
            composer = self.args.composer
            filepath_hidden_state = f'{composer}_hidden_state.npy'
            filepath_cell_state = f'{composer}_cell_state.npy'
            self.logger.log(f'Loading composer initial hidden state from {filepath_hidden_state}')
            self.logger.log(f'Loading composer initial cell state from {filepath_cell_state}')

            hidden_state = np.reshape(np.load(filepath_hidden_state)[0], (1, CONST['LSTM_SIZE']))
            cell_state = np.reshape(np.load(filepath_cell_state)[0], (1, CONST['LSTM_SIZE']))
        else:
            hidden_state = np.zeros((1, CONST['LSTM_SIZE']))
            cell_state = np.zeros((1, CONST['LSTM_SIZE']))

        class State:
            def __init__(self):
                self.c = cell_state
                self.h = hidden_state
        
        return State()