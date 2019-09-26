import os
import torch

import data_utils.normalization as norm
from data_utils import pose_features
from data_utils.datasets import SinglePoseDataset


def test_compute_z_direction():
    hand_palm_1 = torch.tensor([
        [0.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, -1.0, 0.0],
    ])
    hand_palm_2 = torch.tensor([
        [0.0, 0.1, 1.0],
        [1.0, 0.1, 2.0],
        [1.0, 0.1, 1.0],
        [2.0, 0.1, 1.0],
        [1.0, 0.1, 0.0],
        [2.0, 0.1, -1.0],
    ])

    hands = torch.zeros(2, 21, 3, device='cuda')
    hands[0][:6] = hand_palm_1
    hands[1][:6] = hand_palm_2

    true_z_direction_1 = torch.tensor([[0.0, 0.0, -1.0]], device='cuda')

    z_directions = norm.IndividualNormalizer._compute_z_direction(hands)

    assert z_directions.shape == (hands.shape[0], 3)
    assert torch.allclose(z_directions[0], true_z_direction_1)
    assert z_directions[1][1] > 2 * (z_directions[1][0] + z_directions[1][2])


def test_compute_plane_alignment_rot_mat():
    z_directions = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], device='cuda')

    true_rot_mats = torch.stack([
        -torch.eye(3, 3),
        torch.eye(3, 3),
        torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0]
        ])
    ]).cuda()

    rot_mats = norm.IndividualNormalizer._compute_plane_alignment_rot_mat(z_directions)

    assert true_rot_mats.is_same_size(rot_mats)
    assert torch.allclose(rot_mats, true_rot_mats)


def test_compute_inplane_rot_mat():
    x_directions_2d = torch.tensor([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0]

    ], device='cuda')

    true_rot_mats = torch.stack([
        torch.eye(3, 3),
        torch.tensor([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]),
        torch.tensor([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
    ]).cuda()

    rot_mats = norm.IndividualNormalizer._compute_inplane_rot_mat(x_directions_2d)

    assert true_rot_mats.is_same_size(rot_mats)
    assert torch.allclose(rot_mats, true_rot_mats, atol=1e-6)


def test_individual_normalization():
    # The original hand's wrist is set to (0, 0, 0). All bones have length 2.
    # The thumb is pointing in negative z-direction and the pinky in positive z-direction. Their
    # z-components cancel each other in the center point calculation, which makes the rotation
    # easier for this test case. All other fingers coincide, pointing in positive y-direction.
    # The whole hand is located in the y-z-plane.
    # Normalization should apply the following operations in order:
    #   1. Shifted by 1.2 into negative y-direction
    #   2. Scaling by a factor of 0.5
    #   3. Rotation by 90 degrees around the y axis and followed by -90 degrees around the z axis

    pose = torch.zeros(21, 3)
    true_normalized_pose = torch.zeros(21, 3)

    # Wrist
    true_normalized_pose[0] = torch.tensor([-0.6, 0.0, 0.0])

    # Thumb
    thumb_incides = pose_features.finger_indices(0)
    for prev_joint_idx, joint_idx in zip(thumb_incides[:-1], thumb_incides[1:]):
        pose[joint_idx] = pose[prev_joint_idx] + torch.tensor([0.0, 0.0, -2.0])
        next_joint = true_normalized_pose[prev_joint_idx] + torch.tensor([0.0, 1.0, 0.0])
        true_normalized_pose[joint_idx] = next_joint

    # Pinky
    pinky_incides = pose_features.finger_indices(4)
    for prev_joint_idx, joint_idx in zip(pinky_incides[:-1], pinky_incides[1:]):
        pose[joint_idx] = pose[prev_joint_idx] + torch.tensor([0.0, 0.0, 2.0])
        next_joint = true_normalized_pose[prev_joint_idx] + torch.tensor([0.0, -1.0, 0.0])
        true_normalized_pose[joint_idx] = next_joint

    # Rest
    rest_finger_indices = [pose_features.finger_indices(finger_idx) for finger_idx in range(1, 4)]
    rest_finger_indices = torch.stack([torch.tensor(indices) for indices in rest_finger_indices])
    rest_finger_indices = rest_finger_indices.transpose(0, 1)
    for prev_joint_idx_group, joint_idx_group in zip(rest_finger_indices[:-1],
                                                     rest_finger_indices[1:]):
        pose[joint_idx_group] = pose[prev_joint_idx_group] + torch.tensor([0.0, 2.0, 0.0])
        next_joint = true_normalized_pose[prev_joint_idx_group] + torch.tensor([1.0, 0.0, 0.0])
        true_normalized_pose[joint_idx_group] = next_joint

    batch_size = 42
    poses = pose.repeat(batch_size, 1, 1)
    poses = poses.cuda()

    true_normalized_poses = true_normalized_pose.repeat(batch_size, 1, 1)
    true_normalized_poses = true_normalized_poses.cuda()

    true_shifts = torch.zeros(batch_size, 3, device='cuda')
    true_shifts[:, 1] = -1.2

    true_scalings = 0.5 * torch.ones(batch_size, device='cuda')

    true_rotation = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0]
    ], device='cuda')
    true_rotations = true_rotation.repeat(batch_size, 1, 1)

    normalized_poses, params = norm.IndividualNormalizer.normalize_single(poses)

    assert torch.allclose(params['shift'], true_shifts)
    assert torch.allclose(params['scaling'], true_scalings)
    assert torch.allclose(params['rotation'], true_rotations, atol=1e-6)
    assert true_normalized_poses.is_same_size(normalized_poses)
    assert torch.allclose(normalized_poses, true_normalized_poses, atol=1e-6)


def test_normalization_batch_invariance():
    dataset = SinglePoseDataset(os.path.join('unit_test', 'dummy42_labels'))
    batch1 = dataset[:3]
    batch2 = dataset[:2]

    normalized_poses1, norm_params_1 = norm.IndividualNormalizer.normalize_single(batch1.poses)
    normalized_poses2, norm_params_2 = norm.IndividualNormalizer.normalize_single(batch2.poses)

    assert torch.allclose(norm_params_1['shift'][:2], norm_params_2['shift'])
    assert torch.allclose(norm_params_1['scaling'][:2], norm_params_2['scaling'])
    assert torch.allclose(norm_params_1['rotation'][:2], norm_params_2['rotation'])
    assert torch.allclose(normalized_poses1[:2], normalized_poses2)


def test_individual_normalizer_loop():
    batch = SinglePoseDataset(os.path.join('unit_test', 'dummy42_labels'), num_samples=1000,
                              device='cuda')[:]

    normalized_poses, params = norm.IndividualNormalizer.normalize_single(batch.poses)
    denormalized_poses = norm.IndividualNormalizer.denormalize(normalized_poses, params)

    assert batch.poses.is_same_size(denormalized_poses)
    assert torch.allclose(denormalized_poses, batch.poses, atol=1e-2)


def test_view_point_normalizer():
    poses = torch.rand(42, 21, 3)
    labels = torch.rand(42, 21, 3)

    distance_errors_before = torch.norm(poses - labels, dim=2)
    pose_centers_before = torch.mean(poses, dim=1)
    rotated_poses, rotated_labels, params = norm.ViewPointNormalizer.normalize_pair(poses, labels)
    denormalized_poses = norm.ViewPointNormalizer.denormalize(rotated_poses, params)

    distance_errors_after = torch.norm(rotated_poses - rotated_labels, dim=2)
    pose_centers_after = torch.mean(rotated_poses, dim=1)

    assert torch.allclose(pose_centers_before, pose_centers_after)
    assert torch.allclose(distance_errors_before, distance_errors_after)
    assert torch.allclose(poses, denormalized_poses, atol=1e-2)
