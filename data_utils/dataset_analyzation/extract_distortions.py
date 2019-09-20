"""
For a given dataset (with pair of pose and label files available) extract the distortions and save
them to a file.
No normalization is performed.
"""

import os
import torch

from data_utils import datasets

# Config
dataset_name = 'HANDS17_DPREN_SubjClust_train'
########################################################################

dataset = datasets.PairedPoseDataset(dataset_name, use_preset=True)
distortion_vectors = dataset.poses - dataset.labels

torch.save(distortion_vectors, os.path.join('data', 'distortions', '{}.pt'.format(dataset_name)))
