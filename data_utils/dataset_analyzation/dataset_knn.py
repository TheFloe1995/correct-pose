"""
Perform a K-Neirest-Neighbor (KNN) search on a given dataset (only the labels). The output maps to
each sample in the dataset the k closest samples in the same dataset + the respective disparities.
IMPORTANT NOTE: The first (closest) "neighbor" is always the sample itself.
Optionally the poses can be centered individually prior to the KNN search.
"""

import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

from data_utils import datasets

# Config
dataset_name = 'HANDS17_DPREN_SubjClust_train_labels'
k = 16
mode = 'noshift'
n_jobs = 4
########################################################################

output_dir = 'data/knn'
if not os.path.exists(output_dir):
    raise FileNotFoundError('Directory does not exist: {}'.format(output_dir))

dataset = datasets.SinglePoseDataset(dataset_name)
poses = dataset.poses

# Reduce dimensionality for efficiency by selecting only some joints that should be sufficient to
# characterize the pose:
# TMCP, MMCP, PMCP, TTIP, ITIP, MTIP, RTIP, PTIP
joint_indices = [1, 3, 5, 8, 11, 14, 17, 20]

if mode == 'shift':
    # Shift data such that the wrist is at (0, 0, 0)
    poses = poses - poses[:, 0].view(-1, 1, 3)
elif mode == 'noshift':
    # If data is not shifted, also append the wrist as an important joint.
    joint_indices.append(0)
else:
    raise ValueError('Invalid value for mode: {}'.format(mode))

poses_reduced = poses[:, joint_indices]
poses_reduced = poses_reduced.numpy().reshape(-1, 3 * len(joint_indices))

knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean', n_jobs=n_jobs)
knn.fit(poses_reduced)
distances, indices = knn.kneighbors(poses_reduced)

result_dict = {'distances': distances, 'indices': indices}
output_file_path = os.path.join(output_dir, '{}_{}_{}.npy'.format(dataset_name, mode, k))
np.save(output_file_path, result_dict)
