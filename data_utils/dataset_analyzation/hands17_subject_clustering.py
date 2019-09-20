"""
Script to restore subject information from the HANDS 2017 training set. It is known that each sample
in the dataset belongs to one out of 5 subjects. It can be assumed that each subject has a slightly
different hand shape. By running a simple clustering algorithm on the bone lengths the mapping from
subjects to sample indices can be restored.
"""

import os
import torch
import scipy.cluster.vq as scikmeans

from data_utils import datasets


dataset_name = 'HANDS17_DPREN_all'

dataset = datasets.PairedPoseDataset(dataset_name, use_preset=True)
all_labels = dataset[:].labels

wrist_to_tmcp_lengths = torch.norm(all_labels[:, 0] - all_labels[:, 1], dim=1).reshape(-1, 1)

whitened_lengths = scikmeans.whiten(wrist_to_tmcp_lengths)
means, _ = scikmeans.kmeans(whitened_lengths, 5, iter=10)
mapping, _ = scikmeans.vq(whitened_lengths, means)

output_file_path = os.path.join('data', 'clusterings', dataset_name + '_subjects.pt')
torch.save(torch.from_numpy(mapping), output_file_path)
