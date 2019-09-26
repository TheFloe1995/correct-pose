from abc import ABC, abstractmethod
import torch

from data_utils import helper
from data_utils import pose_features


# A normalizer provides an interface to normalize and denormalize a batch of poses.
# The normalization/denormalization is always a deterministic process.
class BaseNormalizer(ABC):
    @classmethod
    @abstractmethod
    def normalize_single(cls, poses):
        pass

    @classmethod
    @abstractmethod
    def normalize_pair(cls, poses, labels):
        pass

    @staticmethod
    @abstractmethod
    def denormalize(poses, params):
        pass

    @classmethod
    def _compute_z_direction(cls, shifted_poses):
        # The pose is assumed to be shifted to a meaningful center point that (approximately) lies
        # on the palm plane.
        device = shifted_poses.device
        palm_bones = shifted_poses[:, :6]
        normal_estimates = torch.zeros(palm_bones.shape[0], 15, 3, device=device)
        idx = 0
        for i, bones_1 in enumerate(palm_bones.transpose(0, 1)[:-1]):
            for bones_2 in palm_bones.transpose(0, 1)[i + 1:]:
                cross = torch.cross(bones_1, bones_2, dim=1)
                norm = torch.norm(cross, dim=1).view(-1, 1)
                mask = torch.isclose(norm, torch.tensor(0.0, device=device), atol=1e-5)
                mask = mask.view(shifted_poses.shape[0])
                normal_estimates[~mask, idx] = cross[~mask] / norm[~mask]
                normal_estimates[mask, idx] = torch.zeros(3, device=device)
                idx += 1

        z_direction = torch.mean(normal_estimates, dim=1)
        return z_direction

    @classmethod
    def _compute_plane_alignment_rot_mat(cls, palm_normals):
        batch_size = palm_normals.shape[0]
        device = palm_normals.device
        rot_mats = torch.zeros(batch_size, 3, 3, device=device)

        # For a single pose, the target vector t is the negative z-direction (0.0, 0.0, -1.0).
        # We seek the rotation matrix R that rotates the given palm normal n onto the target, such
        # that t = R * n.
        target_vecs = torch.tensor([0.0, 0.0, -1.0], device=device).view(1, 3).repeat(batch_size, 1)
        palm_normals = palm_normals / torch.norm(palm_normals, dim=1).view(-1, 1)

        # The vectors t and n span a plane. First find the normal of that plane by computing the
        # cross product.
        plane_normals = torch.cross(palm_normals, target_vecs, dim=1)

        # In that plane the cosine of the angle between t and n can be computed as follows.
        cosines = helper.batch_dot(palm_normals, target_vecs)

        # If t and n are anti-parallel, the solution following this approach is not unique (infinite
        # possible planes). Therefore filter them out with a mask and simply multiply n by -1.
        mask = torch.isclose(cosines, torch.tensor(-1.0, device=device), atol=1e-5)
        rot_mats[mask] = -1.0 * torch.eye(3, device=device)

        # For the rest apply some more magic from https://math.stackexchange.com/questions/180418
        plane_normals = plane_normals[~mask]
        cosines = cosines[~mask]
        cross_prod_mats = torch.zeros(plane_normals.shape[0], 3, 3, device=device)
        cross_prod_mats[:, [1, 0, 0], [2, 2, 1]] = plane_normals
        cross_prod_mats[:, [2, 2, 1], [1, 0, 0]] = plane_normals
        cross_prod_mats[:, [0, 1, 2], [1, 2, 0]] *= -1.0
        cross_prod_mats_sq = torch.bmm(cross_prod_mats, cross_prod_mats)
        rot_mats[~mask] = torch.eye(3, device=device) + cross_prod_mats
        rot_mats[~mask] += cross_prod_mats_sq * (1.0 / (1.0 + cosines)).view(-1, 1, 1)

        return rot_mats

    @classmethod
    def _compute_inplane_rot_mat(cls, x_directions_2d):
        # It is assumed that the z-axis of the pose coordinate frames is already correctly aligned.
        # Rotating the pose inside that plane is therefore a 2D problem.
        batch_size = x_directions_2d.shape[0]
        device = x_directions_2d.device

        target_x_directions_2d = torch.zeros_like(x_directions_2d, device=device)
        target_x_directions_2d[:, 0] = 1.0
        angles = helper.vector_angle_2d(x_directions_2d, target_x_directions_2d)

        inplane_rot_mats = torch.zeros(batch_size, 3, 3, device=device)
        inplane_rot_mats[:, 2, 2] = 1.0
        inplane_rot_mats[:, [0, 1], [0, 1]] = torch.cos(angles).reshape(batch_size, 1)
        inplane_rot_mats[:, 0, 1] = -torch.sin(angles)
        inplane_rot_mats[:, 1, 0] = -inplane_rot_mats[:, 0, 1]

        return inplane_rot_mats


# A no-op placeholder.
class NoNorm(BaseNormalizer):
    @classmethod
    def normalize_single(cls, poses):
        return poses, {'some_param': torch.zeros(poses.shape[0])}

    @classmethod
    def normalize_pair(cls, poses, labels):
        return poses, labels, {'some_param': torch.zeros(poses.shape[0])}

    @staticmethod
    def denormalize(poses, params):
        return poses


# Center all samples such that the overall variance of coordinate values is decreased.
class Shifter(BaseNormalizer):
    @classmethod
    def normalize_single(cls, poses):
        # Shift poses such that the center of mass it at (0, 0, 0).
        # Shift the labels by the same amount.
        shifts = - poses.mean(dim=1).view(-1, 1, 3)
        shifted_poses = poses + shifts

        return shifted_poses, {'shift': shifts}

    @classmethod
    def normalize_pair(cls, poses, labels):
        shifted_poses, params = cls.normalize_single(poses)
        shifted_labels = labels + params['shift']

        return shifted_poses, shifted_labels, params

    @staticmethod
    def denormalize(poses, params):
        return poses - params['shift']


# Rotate poses such that they are always viewed from a similar view point. The position and scale
# of the poses is not changed.
class ViewPointNormalizer(BaseNormalizer):
    @classmethod
    def normalize_single(cls, poses):
        # First compute the hand palm normals. For the computation the data already needs to be
        # centered at some point on the hand palm, which is here defined by W, IMCP, MMCP, RMCP.
        palm_centered_poses = poses - poses[:, [0, 2, 3, 4]].mean(dim=1).view(-1, 1, 3)
        z_directions = cls._compute_z_direction(palm_centered_poses)

        # Rotate the pose such that the normal of the hand palm points into negative z-direction.
        # The normal is approximated by computing the average of all pair wise cross products of the
        # vectors between the origin and the palm joints. The x-axis direction equals the average
        # vector from origin to IMCP, MMCP, RMCP and PMCP (in the previously defined plain).
        # Remember that after the above shifting operation, the origin lies in the palm plane.
        plane_alignment_rot_mats = cls._compute_plane_alignment_rot_mat(z_directions)
        shifts = - poses.mean(dim=1).view(-1, 1, 3)
        centered_poses = poses + shifts
        plain_aligned_poses_t = torch.bmm(plane_alignment_rot_mats, centered_poses.transpose(1, 2))

        x_directions_2d = plain_aligned_poses_t[:, :2, 2:6].mean(dim=2)
        inplane_rot_mats = cls._compute_inplane_rot_mat(x_directions_2d)

        rotated_poses = torch.bmm(inplane_rot_mats, plain_aligned_poses_t).transpose(1, 2)
        rotated_poses -= shifts
        rotations = torch.bmm(inplane_rot_mats, plane_alignment_rot_mats)

        return rotated_poses, {'rotation': rotations, 'shift': shifts}

    @classmethod
    def normalize_pair(cls, poses, labels):
        rotated_poses, params = cls.normalize_single(poses)

        # Labels are rotated around the center of the predicted pose.
        shifted_labels = labels + params['shift']
        rotated_labels = torch.bmm(shifted_labels, params['rotation'].transpose(1, 2))
        rotated_labels -= params['shift']

        return rotated_poses, rotated_labels, params

    @staticmethod
    def denormalize(poses, params):
        centered_poses = poses + params['shift']
        rotated_poses = torch.bmm(centered_poses, params['rotation'])
        rotated_poses -= params['shift']
        return rotated_poses


class GlobalNormalizer(BaseNormalizer):
    def __init__(self, individual_scaling=False):
        self.individual_scaling = individual_scaling

    def normalize_single(self, poses):
        shift = poses.view(-1, 3).mean(dim=0).view(1, 1, 3)
        shifted_poses = poses - shift

        if self.individual_scaling:
            scaling = shifted_poses.norm(dim=2).mean(dim=1).view(-1, 1, 1)
        else:
            scaling = shifted_poses.view(-1, 3).norm(dim=1).mean().view(1)
        scaled_poses = 1.0 / scaling * shifted_poses

        return scaled_poses, {'shift': shift, 'scaling': scaling}

    def normalize_pair(self, poses, labels):
        normalized_poses, params = self.normalize_single(poses)

        shifted_labels = labels - params['shift']
        normalized_labels = 1.0 / params['scaling'] * shifted_labels

        return normalized_poses, normalized_labels, params

    @staticmethod
    def denormalize(poses, params):
        return params['scaling'] * poses + params['shift']


# Deprecation warning: Wasn't used/updated/tested recently.
# This normalizer shifts, rotates and scales each pose (and each label) individually and independent
# from each other to remove any distracting variance. The assumption here is that a pose corrector
# that is just based on the predicted pose of some backbone model (no image evidence) cannot account
# for errors caused by global shift, rotation or scaling. Under this assumption, removing this
# "noise" should not remove any valuable information from the data.
# Be careful in practice: some datasets and/or backbone models don't fulfill this assumption,
# leading to correlations in the data between global errors and absolute positions.
class IndividualNormalizer(BaseNormalizer):
    @classmethod
    def normalize_single(cls, poses):
        batch_size = poses.shape[0]
        device = poses.device

        # Shift data such that the weighted mean of all joints is at (0, 0, 0).
        # The weights are heuristically defined as follows.
        weights = torch.zeros(poses.shape[1], 1, device=device)
        weights[[0, 2, 3, 4]] = 0.2  # W, IMCP, MMCP, RMCP
        weights[[1, 5]] = 0.1  # TMCP, PMCP

        weighted_means = (weights * poses).sum(dim=1)
        shifts = - weighted_means
        shifted_poses = poses + shifts.view(batch_size, 1, -1)

        # Scale data such that the average bone length of the pose is 1.0.
        bone_lengths = pose_features.lengths_of_all_bones(poses)
        mean_bone_length = bone_lengths.view(batch_size, -1).mean(dim=1)
        scalings = 1.0 / mean_bone_length.view(batch_size, 1, 1)
        scaled_poses = shifted_poses * scalings

        # Rotate the pose such that the normal of the hand palm points into negative z-direction.
        # The normal is approximated by computing the average of all pair wise cross products of the
        # vectors between the origin and the palm joints. The x-axis direction equals the average
        # vector from origin to IMCP, MMCP, RMCP and PMCP (in the previously defined plain).
        # Remember that after the above shifting operation, the origin lies in the palm plane.
        z_directions = cls._compute_z_direction(scaled_poses)
        plane_alignment_rot_mats = cls._compute_plane_alignment_rot_mat(z_directions)
        rotated_poses_t = torch.bmm(plane_alignment_rot_mats, scaled_poses.transpose(1, 2))

        x_directions_2d = rotated_poses_t[:, :2, 2:5].mean(dim=2)
        inplane_rot_mats = cls._compute_inplane_rot_mat(x_directions_2d)
        rotated_poses = torch.bmm(inplane_rot_mats, rotated_poses_t).transpose(1, 2)
        rotations = torch.bmm(inplane_rot_mats, plane_alignment_rot_mats)

        return rotated_poses, {'shift': shifts, 'scaling': scalings, 'rotation': rotations}

    @classmethod
    def normalize_pair(cls, poses, labels):
        normalized_poses, params = cls.normalize_single(poses)
        normalized_labels, _ = cls.normalize_single(labels)
        return normalized_poses, normalized_labels, params

    @classmethod
    def denormalize(cls, poses, params):
        denormalized_poses = torch.bmm(poses, params['rotation'])
        denormalized_poses = denormalized_poses / params['scaling'].view(-1, 1, 1)
        denormalized_poses = denormalized_poses - params['shift'].view(-1, 1, 3)
        return denormalized_poses
