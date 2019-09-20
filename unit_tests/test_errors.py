import numpy as np
import torch

from evaluation import errors
from data_utils import pose_features


def test_distance_error():
    batch_size = 42
    poses = torch.ones(batch_size, 21, 3, device='cuda')
    labels = torch.zeros(batch_size, 21, 3, device='cuda')

    true_distance_errors = torch.sqrt(torch.tensor(3.0, device='cuda')) * torch.ones(batch_size, 21,
                                                                                     device='cuda')

    distance_errors = errors.distance_error(poses, labels)

    assert true_distance_errors.is_same_size(distance_errors)
    assert torch.allclose(distance_errors, true_distance_errors)


def test_bone_length_error():
    batch_size = 42
    poses = torch.zeros(batch_size, 21, 3, device='cuda')
    labels = torch.zeros(batch_size, 21, 3, device='cuda')
    for finger_idx in range(5):
        joint_indices = pose_features.finger_indices(finger_idx)
        base_pose = torch.arange(5, device='cuda')
        poses[:, joint_indices] = base_pose.repeat(3, 1).t().type(torch.float32)
        base_label = torch.arange(5, device='cuda')
        labels[:, joint_indices] = -2.0 * base_label.repeat(3, 1).t().type(torch.float32)

    length_error_val = torch.sqrt(torch.tensor(3.0)) - torch.sqrt(torch.tensor(12.0))
    true_bone_length_errors = length_error_val * torch.ones(batch_size, 20, device='cuda')

    bone_length_errors = errors.bone_length_error(poses, labels)

    assert true_bone_length_errors.is_same_size(bone_length_errors)
    assert torch.allclose(bone_length_errors, true_bone_length_errors)


def test_proportion_error():
    batch_size = 42

    # Example hand has bones that all have length 1.0
    poses = torch.zeros(batch_size, 21, 3, device='cuda')
    for finger_idx in range(5):
        joint_indices = pose_features.finger_indices(finger_idx)
        poses[:, joint_indices, 0] = torch.arange(5, device='cuda').type(torch.float32)

    # Displace tip of thumb and index finger in the labels
    labels = poses.clone()
    labels[:, 8, 0] += 2.0
    labels[:, 11, 0] -= 0.5

    true_proportion_error_matrix = torch.zeros(batch_size, 20, 20, device='cuda')
    true_proportion_error_matrix[:, 3] = 2.0
    true_proportion_error_matrix[:, :, 3] = 2.0
    true_proportion_error_matrix[:, 7] = 1.0
    true_proportion_error_matrix[:, :, 7] = 1.0
    true_proportion_error_matrix[:, (3, 7), (7, 3)] = 5.0
    triu_indices = np.triu_indices(true_proportion_error_matrix[0].shape[0], k=1)
    true_proportion_errors = true_proportion_error_matrix[:, triu_indices[0], triu_indices[1]]

    proportion_errors = errors.proportion_error(poses, labels)

    assert true_proportion_errors.is_same_size(proportion_errors)
    assert torch.allclose(proportion_errors, true_proportion_errors)
