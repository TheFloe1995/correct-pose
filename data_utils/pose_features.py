import torch


def finger_indices(finger_idx):
    mcp_idx = finger_idx + 1
    pip_idx = 6 + finger_idx * 3
    return [0] + [mcp_idx] + list(range(pip_idx, pip_idx + 3))


def joints_of_finger(poses, finger_idx):
    return poses[:, finger_indices(finger_idx)]


def joints_of_all_fingers(poses):
    return torch.stack([joints_of_finger(poses, i) for i in range(5)], dim=1)


def bones_of_all_fingers(poses):
    joints = joints_of_all_fingers(poses)
    a_joints = joints[:, :, :4]
    b_joints = joints[:, :, 1:]
    return b_joints - a_joints


def lengths_of_all_bones(poses):
    bones = bones_of_all_fingers(poses)
    return torch.norm(bones, dim=3)
