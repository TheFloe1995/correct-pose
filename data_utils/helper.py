"""
A collection of general helper functions that don't really fit into any of the other modules.
"""

import torch
import numpy as np

from data_utils import pose_features


def batch_dot(vecs_1, vecs_2):
    batch_size = vecs_1.shape[0]
    vector_size = vecs_1.shape[1]
    dotp = torch.bmm(vecs_1.view(batch_size, 1, vector_size),
                     vecs_2.view(batch_size, vector_size, 1))
    return dotp.reshape(-1)


def batch_outer(vecs_1, vecs_2):
    return torch.bmm(vecs_1.unsqueeze(2), vecs_2.unsqueeze(1))


def batch_2x2_det(mat_batch):
    return mat_batch[:, 0, 0] * mat_batch[:, 1, 1] - mat_batch[:, 0, 1] * mat_batch[:, 1, 0]


def vector_angle_2d(vecs_1, vecs_2):
    determinant = batch_2x2_det(torch.stack([vecs_1, vecs_2]).transpose(0, 1))
    dot_product = batch_dot(vecs_1, vecs_2)
    return torch.atan2(determinant, dot_product)


# For a pose with a number of B bones, returns a matrix of shape (B, B) where the element in row i
# and column j is defined as the ration between the lengths of bone i and bone j.
def cross_proportion_matrix(poses):
    bone_lengths = pose_features.lengths_of_all_bones(poses).reshape(poses.shape[0], -1)
    inv_bone_lengths = 1.0 / bone_lengths
    proportion_matrix = batch_outer(bone_lengths, inv_bone_lengths)
    return proportion_matrix


def check_device_compatibility(objects, device):
    for obj in objects:
        if obj.device != device:
            raise ValueError('An object is not stored on the correct device ({})'.format(device))


def print_hyperparameters(hyperparams, keys, indent=0):
    print('\t' * indent + 'The hyperparameters are:')
    for key in keys:
        if key[0] is None:
            val = hyperparams[key[1]]
        else:
            val = hyperparams[key[0]][key[1]]
        print('\t' * indent + '\t{}: \t{}'.format(key[1], val))


def map_func_to_dict(dictionary, func):
    return {key: func(value) for key, value in dictionary.items()}


# The NAIST HANDS2017 data was normalized/cropped in UVD space in a special way. The corresponding
# parameters can be simplified into a scaling factor and a 3D shift vector.
def simplify_naist_denorm_params(params):
    n_params = len(params)
    scalings = torch.zeros(n_params)
    shifts = torch.zeros((n_params, 3))

    for i, param_set in enumerate(params):
        uv_min = param_set[0]
        translation = param_set[1]
        depth_min = param_set[2]
        window_origin = param_set[3][[0, 1]]
        scaling = param_set[4]

        # For some reason some of the coordinates are stored in order vu instead of uv.
        uv_min = np.flip(uv_min)
        translation = np.flip(translation)

        scalings[i] = 1.0 / scaling
        shifts[i, :2] = torch.from_numpy(uv_min + window_origin - (translation / scaling))
        shifts[i, 2] = float(depth_min)

    return scalings, shifts


# For a denormalization of the form orig_pose = norm_pose * S + c, where S is a diagonal matrix
# with diagonal [s, s, 1] and c is a 3D vector, estimate s and c.
def reconstruct_scaling_and_shift(norm_poses, orig_poses):
    # Only 2 joints of each pose are required to reconstruct the parameters.
    # The scaling s is only applied to u and v and is the same for both.
    # It is sufficient to just derive it from the u values.
    u_norm_1 = norm_poses[:, 0, 0]
    u_norm_2 = norm_poses[:, 1, 0]
    u_orig_1 = orig_poses[:, 0, 0]
    u_orig_2 = orig_poses[:, 1, 0]
    s = (u_norm_1 - u_norm_2) / (u_orig_1 - u_orig_2)
    scalings = 1.0 / s

    shifts_u = u_orig_1 - scalings * u_norm_1
    shifts_v = orig_poses[:, 0, 1] - scalings * norm_poses[:, 0, 1]
    shifts_z = orig_poses[:, 0, 2] - norm_poses[:, 0, 2]
    shifts = torch.stack([shifts_u, shifts_v, shifts_z], dim=1)

    return scalings, shifts


def denorm_naist_poses(poses, scalings, shifts):
    denorm_poses = torch.zeros_like(poses)
    denorm_poses[:, :, :2] = poses[:, :, :2] * scalings.reshape(-1, 1, 1)
    denorm_poses[:, :, 2] = poses[:, :, 2]
    denorm_poses += shifts.reshape(-1, 1, 3)
    return denorm_poses


def uvd_to_xyz(poses, camera_params):
    z = poses[:, :, 2]
    x = (poses[:, :, 0] - camera_params['cx']) * z / camera_params['fx']
    y = (poses[:, :, 1] - camera_params['cy']) * z / camera_params['fy']
    return torch.stack([x, y, z], dim=2)


def xyz_to_uvd(poses, camera_params):
    d = poses[:, :, 2]
    u = poses[:, :, 0] * camera_params['fx'] / d + camera_params['cx']
    v = poses[:, :, 1] * camera_params['fy'] / d + camera_params['cy']
    return torch.stack([u, v, d], dim=2)
