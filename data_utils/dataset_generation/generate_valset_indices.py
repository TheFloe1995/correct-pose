"""
Taking an already preprocessed pose dataset, this script generates a list of indices defining the
validation subset of that dataset.
Two generation modes are supported:
    - cluster: Just specify a ratio p and take the last p percent samples as validation set.
    - naive: Start with a previously defined clustering and take the cluster with the
      highest distance error as validation set.
"""

import os
import torch

from data_utils import datasets

# Config
dataset_name = 'HANDS17_DPREN_all'
clustering_name = 'HANDS17_DPREN_all_subjects'
mode = 'naive'
output_name_suffix = '20p'  # Optional
val_set_size_percent = 20  # Only for naive mode
########################################################################

dataset = datasets.PairedPoseDataset(dataset_name, use_preset=True)
all_indices = torch.arange(len(dataset))

if mode == 'naive':
    val_set_size = int(val_set_size_percent / 100 * len(dataset))
    val_set_indices = all_indices[-val_set_size:]

elif mode == 'cluster':
    clustering = torch.load(os.path.join('data', 'clusterings', clustering_name + '.pt'))
    cluster_labels = set(clustering.tolist())
    worst_distance_error = 0.0
    val_set_indices = []
    for cluster_label in cluster_labels:
        cluster_mask = clustering == cluster_label
        cluster_indices = all_indices[cluster_mask]
        cluster_batch = dataset[cluster_indices]
        distance_error = torch.norm(cluster_batch.labels - cluster_batch.poses, dim=2).mean()
        print('Cluster {} with {} samples has distance error {:.3f}'.format(cluster_label,
                                                                            len(cluster_indices),
                                                                            distance_error))
        if distance_error > worst_distance_error:
            worst_distance_error = distance_error
            val_set_indices = cluster_indices
    val_set_size = len(val_set_indices)
else:
    raise ValueError('Invalid mode: {}'.format(mode))

print('Validation set size: {} ({:.2%})'.format(val_set_size, val_set_size / len(dataset)))

output_file_name = '{}_{}_{}.pt'.format(dataset_name, mode, output_name_suffix)
output_file_path = os.path.join('data', 'subset_indices', output_file_name)

torch.save(val_set_indices, output_file_path)
