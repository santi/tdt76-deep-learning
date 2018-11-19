# -*- coding: utf-8 -*-

from __future__ import print_function, division
#import torch
from helpers.datapreparation import visualize_piano_roll, load_all_dataset

#training_data_dir = './datasets/training/piano_roll_fs1'

#print(load_all_dataset_names(training_data_dir))

#print(load_all_dataset(training_data_dir))



import os
# import torch
#from torch.nn import LSTM
import pandas as pd
from glob import glob
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils


class PianoRollDataset():
    def __init__(self, csv_dir, transform=None):
        """
        Args:
            csv_dir (string): Path to the directory containing csv files of piano rolls.
            transform (callable, optional): Optional transform to be applied
                on each sample.
        """
        self.piano_rolls = load_all_dataset(csv_dir)
        self.transform = transform

    def __len__(self):
        return len(self.piano_rolls)

    def __getitem__(self, idx):
        sample = self.piano_rolls[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample

piano_roll_dataset = PianoRollDataset(csv_dir='datasets/training/piano_roll_fs1_subset/')
print(piano_roll_dataset[2])
#for i in range(len(piano_roll_dataset)):
#    print(piano_roll_dataset[i].shape)

visualize_piano_roll(piano_roll_dataset[2])

def target_tensor(piano_roll):
    tensor = piano_roll[1:]
    tensor = np.append(tensor, [np.ones_like(piano_roll[0])], axis=1)
    tensor = np.append(tensor, [np.ones_like(piano_roll[0])], axis=1)
    tensor = np.append(tensor, [np.ones_like(piano_roll[0])], axis=1)
    tensor = np.append(tensor, [np.ones_like(piano_roll[0])], axis=1)
    tensor = np.append(tensor, [np.ones_like(piano_roll[0])], axis=1)
    return tensor
    #return torch.LongTensor(tensor)

a = piano_roll_dataset[0]
print(a)
print(a.shape)

target = target_tensor(a)
print(target.shape)

visualize_piano_roll()

class Generalist(torch.nn.Module):
    def __init__(self, hidden_size, output_size, max_length):
        super(Generalist, self).__init__()
        self.build_model(hidden_size, output_size, max_length)
    
    def forward(self, x):
        pass

    def build_model(self, hidden_size, output_size, max_length):

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length


        self.lstm = LSTM()




















#fig = plt.figure()
"""
for i in range(len(face_dataset)):
    sample = face_dataset[50 + i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(3, 1, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 7:
        plt.show()
        break

n = 62
img_name = face_dataset[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))



def show_landmarks(image, landmarks):
    # Show image with landmarks
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[50 + i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(3, 1, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 7:
        plt.show()
        break
"""