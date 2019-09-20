"""
DEPRECATION WARNING: Not tested/updated recently.
Script to convert joint annotations from the directory structure of the MSRA15 dataset to a binary files,
pickled with torch.
The data is split into a training set and a test/validation set.
"""

import sys
import os
import numpy as np
import torch

dir_name = sys.argv[1]
output_file_name = sys.argv[2]

subject_labels = []
subject_dirs = next(os.walk(dir_name))[1]
for subject_dir in subject_dirs:
    labels = []
    sequence_dirs = next(os.walk(os.path.join(dir_name, subject_dir)))[1]
    for sequence_dir in sequence_dirs:
        label_file_path = os.path.join(dir_name, subject_dir, sequence_dir, 'joint.txt')
        labels.append(np.loadtxt(label_file_path, dtype='float32', skiprows=1))

    subject_labels.append(np.concatenate(labels).reshape(-1, 21, 3))

train_subjects = set(range(9))
test_subjects = {8}
train_subjects = train_subjects - test_subjects

joint_order = [0, 17, 1, 5, 9, 13, 18, 19, 20, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]

training_data = np.concatenate([subject_labels[idx] for idx in train_subjects])
test_data = np.concatenate([subject_labels[idx] for idx in test_subjects])

training_data = training_data[:, joint_order]
training_data[:, :, 1:] *= -1.0
training_data = torch.from_numpy(training_data)

test_data = test_data[:, joint_order]
test_data = torch.from_numpy(test_data)

train_file_name = output_file_name + '_train_labels.pt'
torch.save(training_data, train_file_name)

test_file_name = output_file_name + '_test_labels.pt'
torch.save(test_data, test_file_name)
