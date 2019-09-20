"""
Experimental script to find nearest neighbors for every pose in HANDS 2017 dataset in the MSRA15
dataset to enable some kind of cross dataset error transfer.
"""

import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

from data_utils import datasets

# Config
source_dataset_name = 'MSRA15_labels'
target_dataset_name = 'HANDS17_DPREN_all_labels'
output_name = 'MSRA_to_HANDS_all'
k = 16
mode = 'shift'
########################################################################

output_dir = 'data/knn'
if not os.path.exists(output_dir):
    raise FileNotFoundError('Directory does not exist: {}'.format(output_dir))

source_dataset = datasets.SinglePoseDataset(source_dataset_name)
target_dataset = datasets.SinglePoseDataset(target_dataset_name)

source_poses = source_dataset[:].poses
target_poses = target_dataset[:].poses

if mode == 'shift':
    source_poses = source_poses - source_poses[:, 0].view(-1, 1, 3)
    target_poses = target_poses - target_poses[:, 0].view(-1, 1, 3)
elif mode != 'noshift':
    raise ValueError('Invalid value for mode: {}'.format(mode))

# Reduce dimensionality by selecting only some joints that should be sufficient to characterize the
# pose. Selecting joints that are similarly annotated in both datasets:
# IMCP, PMCP, TPIP, TTIP, ITIP, MTIP, RTIP, PTIP
joint_indices = [2, 5, 6, 8, 11, 14, 17, 20]
source_poses_reduced = source_poses[:, joint_indices]
source_poses_reduced = source_poses_reduced.numpy().reshape(-1, 3 * len(joint_indices))
target_poses_reduced = target_poses[:, joint_indices]
target_poses_reduced = target_poses_reduced.numpy().reshape(-1, 3 * len(joint_indices))

knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean', n_jobs=8)
knn.fit(target_poses_reduced)
distances, indices = knn.kneighbors(source_poses_reduced)

result_dict = {'distances': distances, 'indices': indices}
output_file_path = os.path.join(output_dir, '{}_{}_{}.npy'.format(output_name, mode, k))
np.save(output_file_path, result_dict)
