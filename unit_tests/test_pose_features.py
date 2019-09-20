import torch

from data_utils import pose_features

batch_size = 42
poses_a = torch.zeros(batch_size, 21, 3)
middle_finger_a = torch.tensor([
    [0.0,  0.0,  0.0],
    [1.0,  0.0,  0.0],
    [1.0,  1.0,  0.0],
    [1.0,  1.0,  1.0],
    [0.0, -1.0, -1.0]
])
middle_finger_indices = [0, 3, 12, 13, 14]
poses_a[:, middle_finger_indices] = middle_finger_a


def test_joints_of_finger():
    true_middle_fingers = middle_finger_a.repeat(batch_size, 1, 1)

    middle_fingers = pose_features.joints_of_finger(poses_a, 2)

    assert middle_fingers.is_same_size(true_middle_fingers)
    assert torch.allclose(middle_fingers, true_middle_fingers)


def test_bones_of_all_fingers():
    true_middle_finger_bones = torch.tensor([
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [-1.0, -2.0, -2.0]
    ])
    true_bones_batch = torch.zeros(batch_size, 5, 4, 3)
    true_bones_batch[:, 2] = true_middle_finger_bones

    bones_batch = pose_features.bones_of_all_fingers(poses_a)

    assert true_bones_batch.is_same_size(bones_batch)
    assert torch.allclose(true_bones_batch, bones_batch)


def test_bone_lengths_of_all_fingers():
    true_middle_finger_bone_lengths = torch.tensor([1.0, 1.0, 1.0, 3.0])

    true_lengths_batch = torch.zeros(batch_size, 5, 4)
    true_lengths_batch[:, 2] = true_middle_finger_bone_lengths

    lengths_batch = pose_features.lengths_of_all_bones(poses_a)

    assert true_lengths_batch.is_same_size(lengths_batch)
    assert torch.allclose(true_lengths_batch, lengths_batch)
