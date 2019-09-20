"""
Quick verification whether a newly generated dataset (splitted) looks "reasonable".
"""

import torch

from data_utils import datasets

# Config
dataset_name = 'HANDS17_DPREN_SubjClust'
whole_dataset_name = 'HANDS17_DPREN_all'
########################################################################

train_set = datasets.PairedPoseDataset(dataset_name + '_train', use_preset=True)
val_set = datasets.PairedPoseDataset(dataset_name + '_val', use_preset=True)
whole_set = datasets.PairedPoseDataset(whole_dataset_name, use_preset=True)

train_distance_error = torch.norm(train_set[:].poses - train_set[:].labels, dim=2).mean()
val_distance_error = torch.norm(val_set[:].poses - val_set[:].labels, dim=2).mean()

assert len(train_set) + len(val_set) == len(whole_set)
assert train_distance_error > 1e-3
assert val_distance_error > 1e-3
assert train_distance_error != val_distance_error
assert not torch.allclose(train_set[0].labels, val_set[0].labels)

print('Looks good!')
