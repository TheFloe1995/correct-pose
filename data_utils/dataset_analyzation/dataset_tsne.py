"""
DEPRECATION WARNING: Not used/updated/tested recently.
Script to perform a TSNE embedding on a given dataset.
"""

import os
import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE

from data_utils import datasets
from data_utils import distorters
from data_utils import pose_features
from evaluation import errors


# Helper Functions
def joint_subset(poses):
    subset_indices = [0, 1, 3, 5, 8, 11, 14, 17, 20]
    return poses[:, subset_indices]


def combined_errors(poses, labels):
    mean_distance_errors = errors.distance_error(poses, labels).mean(dim=1)
    mean_bone_length_errors = errors.bone_length_error(poses, labels).mean(dim=1)
    distance_between_centers = torch.norm(poses.mean(dim=1) - labels.mean(dim=1), dim=1)

    print('Computing covariances...')
    pose_cov_mats  = np.array([np.cov(pose.numpy()) for pose in poses])
    pose_cov_mats = torch.from_numpy(pose_cov_mats).type(torch.float32)
    label_cov_mats = np.array([np.cov(label.numpy()) for label in labels])
    label_cov_mats = torch.from_numpy(label_cov_mats).type(torch.float32)
    distance_between_covariances = torch.norm(pose_cov_mats - label_cov_mats, dim=(1, 2))

    combined = torch.stack([mean_distance_errors,
                            mean_bone_length_errors,
                            distance_between_centers,
                            distance_between_covariances],
                           dim=1)

    combined = combined / combined.mean(dim=0)
    return combined

########################################################################

# Config
dataset_names = ['HANDS17_DPREN_ShapeSplit_val']
eval_spaces = ['original']
subset_name = 'DEFAULT'
error_name = 'dist_disp'
perplexities = [30, 50, 70]
cont = False
max_iter = 2500
cheat_metric = False

print('Starting TSNE analysis with: \n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(
    dataset_names, eval_spaces,
    subset_name, error_name,
    perplexities, cont,
    max_iter, cheat_metric))

########################################################################

for dataset_name in dataset_names:
    for eval_space in eval_spaces:
        for perplexity in perplexities:
            if eval_space == 'normalized':
                dataset = datasets.NormalizedPairedPoseDataset(dataset_name,
                                                               distorters.NoDistorter(),
                                                               True)
            else:
                dataset = datasets.PairedPoseDataset(dataset_name, distorters.NoDistorter(), True)
            dataset.select_subset(subset_name)

            all_data = dataset[:]

            if error_name == 'distance':
                data = errors.distance_error(all_data.poses, all_data.labels)
            elif error_name == 'bone_length':
                data = errors.bone_length_error(all_data.poses, all_data.labels)
            elif error_name == 'dist_bone_cat':
                distance_errors = errors.distance_error(all_data.poses, all_data.labels)
                bone_length_errors = errors.bone_length_error(all_data.poses, all_data.labels)
                data = torch.cat((distance_errors, bone_length_errors), dim=1)
            elif error_name == 'poses_only':
                data = all_data.poses.reshape(-1, 63)
            elif error_name == 'combined':
                data = combined_errors(all_data.poses, all_data.labels)
            elif error_name == 'shape':
                data = pose_features.lengths_of_all_bones(all_data.labels).reshape(-1, 20)
            elif error_name == 'dist_disp':
                distance_errors = errors.distance_error(all_data.poses, all_data.labels)
                disparities = np.load(os.path.join('results', dataset_name + '_disparities.npy'))
                disparities = disparities.reshape(-1, 1)
                data = torch.cat((distance_errors,
                                  torch.from_numpy(disparities).type(torch.float32)),
                                 dim=1)
            else:
                raise ValueError('Unknown error function name: {}'.format(error_name))

            data = data / torch.mean(data, dim=0)

            file_name = os.path.join('results/00_TSNE', '{}_{}_{}_{}_{}'.format(dataset_name,
                                                                                eval_space,
                                                                                subset_name,
                                                                                error_name,
                                                                                perplexity))

            if cont:
                precomputed_embedding = np.load(file_name + '.npy')
                tsne = TSNE(n_components=2, verbose=3, n_jobs=6, perplexity=perplexity,
                            n_iter=max_iter, cheat_metric=cheat_metric, init=precomputed_embedding)
            else:
                tsne = TSNE(n_components=2, verbose=3, n_jobs=6, perplexity=perplexity,
                            n_iter=max_iter, cheat_metric=cheat_metric)

            embedded_errors = tsne.fit_transform(data.numpy())
            np.save(file_name + '.npy', embedded_errors)
            print('FINISH: {}'.format(file_name))
