"""
Many authors publish the MSRA15 test results in a single file for the whole dataset (tested in a
leave-one-out cross validation scheme over the 9 subjects). This script prints the indices at which
samples taken from a new subject start (they are saved in a row).
"""

import os
import numpy as np

dir_name = 'data/MSRA15'

subject_labels = []
subject_dirs = next(os.walk(dir_name))[1]
for subject_dir in subject_dirs:
    labels = []
    sequence_dirs = next(os.walk(os.path.join(dir_name, subject_dir)))[1]
    for sequence_dir in sequence_dirs:
        label_file_path = os.path.join(dir_name, subject_dir, sequence_dir, 'joint.txt')
        labels.append(np.loadtxt(label_file_path, dtype='float32', skiprows=1))

    subject_labels.append(np.concatenate(labels).reshape(-1, 21, 3))

start_idx = 0
for i, labels in enumerate(subject_labels):
    print('Subject {} has {} samples, starting at index: {}'.format(i, labels.shape[0], start_idx))
    start_idx += labels.shape[0]
