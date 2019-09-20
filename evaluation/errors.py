import torch
import numpy as np

from data_utils import helper
from data_utils import pose_features


# Simply take the difference between all coordinate pairs of all joints.
# The result is of shape (N, J * D)
def coordinate_difference(poses, labels):
    return (poses - labels).reshape(poses.shape[0], -1)


# The euclidean distance between each joint pair.
# The result is of shape (N, J)
def distance_error(poses, labels):
    return torch.norm(coordinate_difference(poses, labels).view(*poses.shape), dim=2)


# The difference in length between each bone pair.
# The result is of shape (N, B)
def bone_length_error(poses, labels):
    bone_lengths = pose_features.lengths_of_all_bones(poses)
    true_bone_lengths = pose_features.lengths_of_all_bones(labels)
    return (bone_lengths - true_bone_lengths).reshape(poses.shape[0], -1)


# The differences between the proportions of pose and label. The proportions are the "symmetric
# pairwise length ratios" between all bones of a single pose.
def proportion_error(poses, labels):
    # First compute the proportion matrices of shape (B, B).
    proportion_matrices = helper.cross_proportion_matrix(poses)
    true_proportion_matrices = helper.cross_proportion_matrix(labels)

    # Only one triangle of the matrix is relevant because the other triangle just contains the
    # inverse values. The diagonal is irrelevant because all elements are always 1.0.
    # Note that recent versions of PyTorch also provide a triu_indices function but for
    # compatibility reasons, numpy is used here.
    triu_idx = np.triu_indices(proportion_matrices[0].shape[0], k=1)
    relevant_props = proportion_matrices[:, triu_idx[0], triu_idx[1]]
    true_relevant_props = true_proportion_matrices[:, triu_idx[0], triu_idx[1]]

    # Making the error symmetric.
    errors = torch.zeros(poses.shape[0], relevant_props.shape[1], device=poses.device)
    mask = relevant_props > true_relevant_props
    errors[mask] = relevant_props[mask] / true_relevant_props[mask] - 1.0
    errors[~mask] = true_relevant_props[~mask] / relevant_props[~mask] - 1.0

    # Clip errors to a maximum of 10.0. Everything above leads to very high squared terms which is
    # semantically not very useful. Everything above 10.0 is therefore considered to be
    # "equally very wrong".
    errors = torch.clamp(errors, 0.0, 10.0)

    return errors
