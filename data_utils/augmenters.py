from abc import ABC, abstractmethod
import torch
import torch.distributions as torch_dist
import scipy.stats


class BaseAugmenter(ABC):
    @abstractmethod
    def augment(self, batch):
        pass

    @staticmethod
    def _center_batch(batch):
        pose_shifts = batch.poses.mean(dim=1).view(-1, 1, 3)
        label_shifts = batch.labels.mean(dim=1).view(-1, 1, 3)
        centered_poses = batch.poses - pose_shifts
        centered_labels = batch.labels - label_shifts
        return centered_poses, centered_labels, pose_shifts, label_shifts


# Randomly scales pose and label in place by the same factor
class RandomScaler(BaseAugmenter):
    def __init__(self, scale_std):
        self.scale_sampler = torch_dist.Normal(1.0, scale_std)

    def augment(self, batch):
        scaling_factors = self.scale_sampler.sample((len(batch), 1, 1)).to(batch.device)

        centered_poses, centered_labels, pose_shifts, label_shifts = self._center_batch(batch)
        centered_poses *= scaling_factors
        centered_labels *= scaling_factors

        batch.poses = centered_poses + pose_shifts
        batch.labels = centered_labels + label_shifts


# Applies a different random 3D rotation to each sample pair in the batch.
class RandomRotator(BaseAugmenter):
    def augment(self, batch):
        pose_shifts = batch.poses.mean(dim=1).view(-1, 1, 3)
        label_shifts = batch.labels.mean(dim=1).view(-1, 1, 3)
        centered_poses = batch.poses - pose_shifts
        centered_labels = batch.labels - label_shifts

        rotation_matrics = scipy.stats.special_ortho_group.rvs(3, size=len(batch)).astype('float32')
        rotation_matrices = torch.from_numpy(rotation_matrics).to(batch.device)
        rotated_poses = torch.bmm(centered_poses, rotation_matrices)
        rotated_labels = torch.bmm(centered_labels, rotation_matrices)

        batch.poses = rotated_poses + pose_shifts
        batch.labels = rotated_labels + label_shifts


# Samples a new pose by interpolating between the given input pose and the label. Extrapolation is
# also supported.
# In detail the operation is x' = k * e + y, where e = x - y. For k=1 the original pair (x, y) is
# obtained. For k=0 the new pose x' is equal to the label.
# TODO: This is actually a distortion, not an augmentation (see README). --> Move to distorters.
class RandomInterpolator:
    def __init__(self, alpha, loc, scale):
        self.sampler_params = (alpha, loc, scale)
        self.random_factor_sampler = scipy.stats.skewnorm(a=alpha, loc=loc, scale=scale)

    def augment(self, batch):
        joint_displacements = batch.poses - batch.labels
        interpolation_factors = self.random_factor_sampler.rvs(size=len(batch)).astype('float32')
        interpolation_factors = torch.from_numpy(interpolation_factors).to(batch.device).view(-1, 1, 1)
        batch.poses = batch.labels + joint_displacements * interpolation_factors

    def __str__(self):
        return str(self.sampler_params)
