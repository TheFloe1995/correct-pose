import os
from abc import ABC, abstractmethod
import torch
import numpy as np
from torch import distributions
import scipy.stats

# This crazy matrix defines the probabilities for each joint (row) to be confused with another joint
# (col) given that a confusion happened.
# TODO: Outsource this into a separate file from which it can be loaded, but keep it readable.
confusion_target_probabilities = torch.tensor([
    #W,  TMCP, IMCP, MMCP, RMCP, PMCP, TPIP, TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # W
    [0.0, 0.0, 0.25, 0.25, 0.25, 0.25,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # TMCP
    [0.0, 0.2, 0.0,  0.5,  0.2,  0.1,   0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # IMCP
    [0.0, 0.2, 0.35, 0.0,  0.35, 0.1,   0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # MMCP
    [0.0, 0.2, 0.1,  0.35, 0.0,  0.35,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # RMCP
    [0.0, 0.2, 0.1,  0.2,  0.5,  0.0,   0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # PMCP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.0, 0.25, 0.0,  0.0,  0.25, 0.0,  0.0,  0.25, 0.0,  0.0,  0.25, 0.0,  0.0],  # TPIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.0, 0.0,  0.25, 0.0,  0.0,  0.25, 0.0,  0.0,  0.25, 0.0,  0.0,  0.25, 0.0],  # TDIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.0, 0.0,  0.0,  0.25, 0.0,  0.0,  0.25, 0.0,  0.0,  0.25, 0.0,  0.0,  0.25], # TTIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.2,  0.0,  0.0, 0.0,  0.0,  0.0,  0.5,  0.0,  0.0,  0.2,  0.0,  0.0,  0.1,  0.0,  0.0],  # IPIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.2,  0.0, 0.0,  0.0,  0.0,  0.0,  0.5,  0.0,  0.0,  0.2,  0.0,  0.0,  0.1,  0.0],  # IDIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.2, 0.0,  0.0,  0.0,  0.0,  0.0,  0.5,  0.0,  0.0,  0.2,  0.0,  0.0,  0.1],  # ITIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.2,  0.0,  0.0, 0.35, 0.0,  0.0,  0.0,  0.0,  0.0,  0.35, 0.0,  0.0,  0.1,  0.0,  0.0],  # MPIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.2,  0.0, 0.0,  0.35, 0.0,  0.0,  0.0,  0.0,  0.0,  0.35, 0.0,  0.0,  0.1,  0.0],  # MDIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.2, 0.0,  0.0,  0.35, 0.0,  0.0,  0.0,  0.0,  0.0,  0.35, 0.0,  0.0,  0.1],  # MTIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.2,  0.0,  0.0, 0.1,  0.0,  0.0,  0.35, 0.0,  0.0,  0.0,  0.0,  0.0,  0.35, 0.0,  0.0],  # RPIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.2,  0.0, 0.0,  0.1,  0.0,  0.0,  0.35, 0.0,  0.0,  0.0,  0.0,  0.0,  0.35, 0.0],  # RDIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.2, 0.0,  0.0,  0.1,  0.0,  0.0,  0.35, 0.0,  0.0,  0.0,  0.0,  0.0,  0.35], # RTIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.2,  0.0,  0.0, 0.1,  0.0,  0.0,  0.2,  0.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.0],  # PPIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.2,  0.0, 0.0,  0.1,  0.0,  0.0,  0.2,  0.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0],  # PDIP
    [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.2, 0.0,  0.0,  0.1,  0.0,  0.0,  0.2,  0.0,  0.0,  0.5,  0.0,  0.0,  0.0],  # PTIP
])


class BaseDistorter(ABC):
    def __call__(self, poses, indices=None):
        return self.distort(poses, indices)

    @abstractmethod
    def distort(self, poses, indices=None):
        pass


# A no-op placeholder.
class NoDistorter(BaseDistorter):
    def __init__(self, *args):
        pass

    def distort(self, poses, indices=None):
        return poses


# Distorter that applies errors sampled from a heuristically defined distribution to the data.
# Config:
#   confusion_prob: float, probability for a joint (except the wrist) to be confused with another
#                   joint
#   layer_probs: list of floats, probabilities from which layer the distortion should be sampled if
#                a jitter error applies
#   stds: list of floats, standard deviations of each layers Gaussian
#   layer_radii: list of floats, radii of the spherical shells (layers)
class SyntheticDistorter(BaseDistorter):
    def __init__(self, config):
        self.config = config
        self.confusion_target_probs = confusion_target_probabilities
        self.confusion_probs = torch.tensor([0.0] + [config['confusion_prob']] * 20)

    def distort(self, poses, indices=None):
        device = poses.device
        distorted_hands = torch.zeros_like(poses)

        # Sample for each pose the joints that should receive a confusion error. A confusion means
        # that a joint is placed (close) to the position of another joint as if they were confused
        # by some imaginary predictor.
        # Therefore a mask of shape (data_set_size, 21) is sampled from a Bernoulli distribution.
        confusion_probs = self.confusion_probs.repeat(poses.shape[0], 1).to(device)
        confusion_mask = torch.bernoulli(confusion_probs).byte().to(device)

        distorted_hands[confusion_mask] = self._confuse(confusion_mask, poses)

        # Apply some jitter to all other samples.
        distorted_hands[~confusion_mask] = self._layered_gaussian_noise(poses[~confusion_mask])

        return distorted_hands

    def _layered_gaussian_noise(self, joints):
        # Noise is applied in form of 3D displacement vectors e such that the new pose is defined as
        # p' = p + e. The vectors e are sampled from a layered Gaussian distribution which is the
        # same for all joints.
        # Layered means that multiple 3D Gaussians are defined. First one of the Gaussians is
        # selected by sampling from a categorical distribution. Each Gaussian can have a different
        # variance (diagonal covariance matrix with single value assumed) but it is always centered
        # at the original joint position p. However, points sampled from outer layers are always
        # displaced by a constant into the direction of e. Visually this means that sampled joint
        # positions p' are distributed outside the surface of a sphere.
        device = joints.device

        displacement_layers = torch.multinomial(torch.tensor(self.config['layer_probs']),
                                                joints.shape[0], replacement=True)
        displacement_layers = displacement_layers.flatten().to(device)
        distorted_joints = torch.zeros_like(joints)

        for i, (std, r) in enumerate(zip(self.config['stds'], self.config['layer_radii'])):
            indices = torch.nonzero(displacement_layers == i).flatten()

            distortion_vectors = torch.normal(torch.zeros(indices.numel(), joints.shape[1]), std)
            distortion_vectors = distortion_vectors.to(device)
            distortion_directions = distortion_vectors / distortion_vectors.norm(dim=1).unsqueeze(1)
            distortions = r * distortion_directions + distortion_vectors

            distorted_joints[indices] = joints[indices] + distortions
        return distorted_joints

    def _confuse(self, mask, hands):
        # Each row of the confusion probability matrix corresponds to a joint i and specifies the
        # conditional probability (given that a confusion error applies to i) of i being confused
        # with target joint j (column).
        confusion_joint_indices = torch.nonzero(mask)
        flat_joint_indices = confusion_joint_indices[:, 1]
        target_probs = self.confusion_target_probs[flat_joint_indices]
        flat_target_indices = torch.multinomial(target_probs, 1).squeeze().to(hands.device)
        target_joints = hands[confusion_joint_indices[:, 0], flat_target_indices]

        # The joint coordinates are just replaced by the target joint's position + Gaussian noise.
        distorted_joints = torch.normal(target_joints, self.config['stds'][0])

        return distorted_joints


# Distortions are not generated at runtime but loaded from a file where each sample can only have
# a single associated distortion.
# To add more randomness the distortion vectors are scaled by a randomly sampled strength factor.
# Config:
#   source_name: str, name of the file without file extension and without containing directory from
#                which to load the distortions
#   strength_mean: float, mean of the Gaussian from which to sample the strength factor
#   strength_std: float, standard deviation of the Gaussian from which to sample the strength factor
#   device: torch.device, to which the distortions should be preloaded (for efficiency)
class PredefinedDistorter(BaseDistorter):
    def __init__(self, config):
        self.config = config
        distortions_file_path = os.path.join('data', 'distortions', config['source_name'] + '.pt')
        self.distortions = torch.load(distortions_file_path)
        self.distortions = self.distortions.to(self.config['device'])
        self.strenth_distribution = distributions.Normal(config['strength_mean'],
                                                         config['strength_std'])

    def distort(self, poses, indices=None):
        indices = torch.randint(len(self.distortions), (len(poses), ), device=self.config['device'])
        strength = self.strenth_distribution.sample().to(self.config['device'])
        distortions = strength * self.distortions[indices]
        return poses + distortions


# Distortions are not generated at runtime but loaded from a predefined file.
# A KNN table (also predefined) maps every sample in the dataset a set of k available distortions.
# The maximum amount of k can be further decreased by the parameter max_k.
# At runtime, one of the k distortions is selected at random (from the neighbor_distribution) and
# then applied to the input pose.
# To add more randomness the distortion vectors are scaled by a randomly sampled strength factor.
# Config:
#   source_name: str, name of the file without file extension and without containing directory from
#                which to load the distortions
#   knn_name: str, name of the file without file extension and without containing directory from
#             which to load the knn table
#   max_k: int >= 1, how many neighbors should be used at maximum
#   strength_alpha|loc|scale: all floats, parameters for the distribution from which to sample the
#                             strength factor (see scipy.stats.skewnorm for details)
class KNNPredefinedDistorter(BaseDistorter):
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        distortions_file_path = os.path.join('data', 'distortions', config['source_name'] + '.pt')
        self.distortions = torch.load(distortions_file_path, map_location=self.device)
        knn_results = np.load(os.path.join('data', 'knn', config['knn_name'] + '.npy')).item()
        self.knn_indices = torch.from_numpy(knn_results['indices']).to(self.device)
        self.knn_distances = torch.from_numpy(knn_results['distances']).to(self.device)
        self.max_k = self.config['max_k']

        self.strenth_distribution = scipy.stats.skewnorm(a=config['strength_alpha'],
                                                         loc=config['strength_loc'],
                                                         scale=config['strength_scale'])
        neighbor_distrib_probs = torch.linspace(100, 10, self.max_k)
        self.neighbor_distribution = distributions.Categorical(neighbor_distrib_probs)

    def distort(self, poses, indices=slice(None)):
        n_poses = len(poses)
        knn_indices = self.knn_indices[indices, :self.max_k].view(n_poses, -1)
        selection = self.neighbor_distribution.sample((n_poses,))
        distortion_indices = knn_indices[torch.arange(n_poses), selection]

        strengths = self.strenth_distribution.rvs(size=n_poses).astype('float32')
        strengths = torch.from_numpy(strengths).to(self.device).view(-1, 1, 1)
        distortions = strengths * self.distortions[distortion_indices]
        return poses + distortions
