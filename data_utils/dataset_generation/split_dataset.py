"""
Split a dataset into a training set and a validation set using predefined indices for the
validation set.
"""

import os
import torch

from data_utils import datasets

# Config
dataset_name = 'HANDS17_DPREN_all'
val_indices_file_name = 'HANDS17_DPREN_all_cluster_subjects'
output_name = 'HANDS17_DPREN_SubjClust'
########################################################################

dataset = datasets.PairedPoseDataset(dataset_name, use_preset=True)

val_indices_file_path = os.path.join('data', 'subset_indices', val_indices_file_name + '.pt')
val_indices = torch.load(val_indices_file_path)

train_set_mask = torch.ones(len(dataset), dtype=torch.uint8)
train_set_mask[val_indices] = 0

train_batch = dataset[train_set_mask]
val_batch = dataset[val_indices]

output_path_prefix = os.path.join('data', output_name)

torch.save(train_batch.poses, '{}_train_poses.pt'.format(output_path_prefix))
torch.save(train_batch.labels, '{}_train_labels.pt'.format(output_path_prefix))
torch.save(val_batch.poses, '{}_val_poses.pt'.format(output_path_prefix))
torch.save(val_batch.labels, '{}_val_labels.pt'.format(output_path_prefix))
