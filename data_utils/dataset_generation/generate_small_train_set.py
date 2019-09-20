"""
Artificially extract a smaller training set from a big one for experiments that simulate a lack of
data.
"""

import os
import torch

from data_utils import datasets

# Config
dataset_name = 'HANDS17_DPREN_SubjClust'
small_set_size_percent = 10
########################################################################

dataset = datasets.PairedPoseDataset(dataset_name + '_train', use_preset=True)

small_set_size = int(small_set_size_percent / 100 * len(dataset))
subset_batch = dataset[:small_set_size_percent]

output_name = '{}_small{}_train'.format(dataset_name, small_set_size_percent)
torch.save(subset_batch.poses, os.path.join('data', output_name + '_poses.pt'))
torch.save(subset_batch.labels, os.path.join('data', output_name + '_labels.pt'))
