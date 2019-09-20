from abc import ABC
import os
import torch
from torch.utils.data import Dataset, RandomSampler, BatchSampler, SequentialSampler

from data_utils import normalization as norm
from data_utils import distorters


# A data structure that encapsules all the relevant data needed during training for a single batch.
# It can hold just a single set of poses or a paired set of poses and labels. If the data is
# normalized it additionally provides access to the original (i.e. without normalization) data and
# the parameters used for normalization.
class PoseCorrectionBatch:
    def __init__(self, poses, labels=None, original_poses=None, original_labels=None,
                 normalization_params=None):
        self.poses = poses
        self.device = poses.device
        self.labels = None
        self.original_poses = None
        self.original_labels = None

        if labels is not None:
            self.labels = labels

        if original_poses is None or original_labels is None:
            if original_poses != original_labels:
                raise ValueError('Unsupported sample batch initialization. ' +
                                 'Either provide both original labels and poses or nothing.')
            self.normalized = False
        else:
            self.original_poses = original_poses
            self.original_labels = original_labels
            self.normalization_params = normalization_params
            self.normalized = True

    def __len__(self):
        return self.poses.shape[0]

    # Returns a list of all contained tensors.
    # TODO: Implement this as an iterator.
    def list_tensors(self):
        tensor_list = [self.poses]
        if self.labels is not None:
            tensor_list.append(self.labels)
        if self.original_poses is not None:
            tensor_list.append(self.original_poses)
            tensor_list.append(self.original_labels)
        return tensor_list


# For a general description of the dataset format, see the README. It is an iterable and samples can
# be accessed using the [] operator, where a PoseCorrectionBatch is returned.
# A pose dataset is always stored on a single device.
# It contains at least a set of poses and can have subsets. There is always only a single active
# subset. Calling the default data getter returns only samples from the active set. To get samples
# from different subsets the active subset needs to be switched.
class BaseDataset(Dataset, ABC):
    def __init__(self, name, device='cpu'):
        self.name = name
        self.device = torch.device(device)
        self.n_samples = None
        self.poses = None
        self.pose_subsets = None
        self.current_subset = 'DEFAULT'

    def __len__(self):
        if self.n_samples is None:
            if type(self.poses) is dict:
                self.n_samples = self._len_from_dict(self.poses)
            else:
                self.n_samples = self.poses.shape[0]
        return self.n_samples

    @property
    def has_subsets(self):
        return self.pose_subsets is not None

    def get_subset_names(self):
        try:
            return list(sorted(self.pose_subsets.keys()))
        except AttributeError:
            raise TypeError('This dataset does not have subsets.')

    # Select a different subset by specifying the desired name.
    def select_subset(self, subset_name):
        try:
            if not subset_name == 'DEFAULT':
                self.poses = self.pose_subsets[subset_name]
            self.current_subset = subset_name
        except TypeError:
            raise TypeError('This dataset does not have subsets.')
        except KeyError:
            raise KeyError('The subset "{}" doesn\'nt exist in the dataset'.format(subset_name))

    @staticmethod
    def _load_from_disk(full_name, device):
        poses_file = os.path.join('data', full_name + '.pt')
        return torch.load(poses_file, map_location=device)

    @staticmethod
    def _len_from_dict(data_dict):
        return next(iter(data_dict.values())).shape[0]

    @classmethod
    def _shorten_and_to_device(cls, data, num_samples, device):
        if type(data) is dict:
            if num_samples is None:
                num_samples = cls._len_from_dict(data)
            return {key: poses[:num_samples].to(device) for key, poses in data.items()}
        else:
            if num_samples is None:
                num_samples = data.shape[0]
            return data[:num_samples].to(device)

    # The returned data must have 3 dimensions (N, J, D), even if just a single sample is queried.
    @staticmethod
    def _batchify(pose_tensor):
        return pose_tensor.view(-1, 21, 3)


# For visualization and some analyzation applications only a single set of poses is
# needed without a paired set of labels.
class SinglePoseDataset(BaseDataset):
    def __init__(self, name, normalizer=norm.NoNorm, num_samples=None, device='cpu'):
        super().__init__(name, device)

        poses = self._load_from_disk(name, device)

        if type(poses) is dict:
            self.pose_subsets = self._shorten_and_to_device(poses, num_samples, device)
            self.normalization_params = {}
            for subset_name in poses.keys():
                normalization_results = normalizer.normalize_single(self.pose_subsets[subset_name])
                poses[subset_name] = normalization_results[0]
                self.normalization_params[subset_name] = normalization_results[1]
            self.select_subset(self.get_subset_names()[0])
        else:
            self.poses = self._shorten_and_to_device(poses, num_samples, device)
            poses, self.normalization_params = normalizer.normalize_single(poses)

    def __getitem__(self, idx):
        return PoseCorrectionBatch(self._batchify(self.poses[idx]))


# Standard dataset that returns a PoseCorrectionBatch containing input poses and labels.
# The constructor accepts a distorter object that is applied to the pose when a sample is accessed
# from outside.
# When use_preset==True, the poses are loaded from a separate file from disk.
# When use_preset==False, the poses are set equal to the labels internally. This setting makes only
# sense if a distorter other than NoDistorter is used. In this case the poses are generated from the
# labels by applying an error (distortion).
class PairedPoseDataset(BaseDataset):
    def __init__(self, name, distorter=distorters.NoDistorter(), use_preset=False, num_samples=None,
                 device='cpu'):
        super().__init__(name + '_labels', device)

        self.distorter = distorter

        label_data = self._load_from_disk(name + '_labels', device)
        if type(label_data) is dict:
            raise TypeError('Subsets for labels are not supported.')
        self.labels = self._shorten_and_to_device(label_data, num_samples, device)

        self.use_preset = use_preset
        if use_preset:
            pose_data = self._load_from_disk(name + '_poses', device)
            if type(pose_data) is dict:
                self.pose_subsets = self._shorten_and_to_device(pose_data, num_samples, device)
                self.select_subset(self.get_subset_names()[0])
            else:
                self.poses = self._shorten_and_to_device(pose_data, num_samples, device)
        else:
            self.poses = self.labels

        if len(self.poses) != len(self.labels):
            raise RuntimeError('Different number of poses and labels.')

    def __getitem__(self, idx):
        distorted_poses = self.distorter(self._batchify(self.poses[idx]), idx)
        batch = PoseCorrectionBatch(distorted_poses, self._batchify(self.labels[idx]))
        return batch


# Same functionality as PairedPoseDataset but adds normalization. If the default normalizer NoNorm
# is used, the behavior is equal to PairedPoseDataset but a small overhead is added.
# TODO: How significant is the overhead? Maybe it's better to merge with super class.
class NormalizedPairedPoseDataset(PairedPoseDataset):
    def __init__(self, name, distorter=distorters.NoDistorter(), normalizer=norm.NoNorm,
                 use_preset=False, num_samples=None, device='cpu'):
        super().__init__(name, distorter, use_preset, num_samples, device)

        poses = self._batchify(self.poses)
        labels = self._batchify(self.labels)
        self.normalizer = normalizer
        self.norm_poses, self.norm_labels, self.norm_params = normalizer.normalize_pair(poses,
                                                                                        labels)

    def __getitem__(self, idx):
        distorted_poses = self.distorter(self._batchify(self.norm_poses[idx]), idx)
        return PoseCorrectionBatch(distorted_poses,
                                   self._batchify(self.norm_labels[idx]),
                                   self._batchify(self.poses[idx]),
                                   self._batchify(self.labels[idx]),
                                   self._index_norm_params_dict(self.norm_params, idx))

    @staticmethod
    def _index_norm_params_dict(norm_params_dict, idx):
        result_dict = {}
        for key, params in norm_params_dict.items():
            if params.shape[0] > 1:
                result_dict[key] = params[idx]
            else:
                result_dict[key] = params
        return result_dict


# This class provides a similar interface as torch.utils.data.DataLoader (which cannot be used here
# because of the special datatypes used. Basically all the logic is wrapped for randomly sampling
# mini batches from a dataset of a specific batch size. It also allows to randomly switch subsets
# during training.
class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, shuffle_subsets=False):
        self.dataset = dataset
        if shuffle:
            core_sampler = RandomSampler(dataset)
        else:
            core_sampler = SequentialSampler(dataset)
        self.sampler = BatchSampler(core_sampler, batch_size, drop_last=False)
        self.n_batches = int(torch.ceil(torch.tensor(len(dataset) / batch_size)))
        self.sampler_iterator = None
        self.shuffle_subsets = shuffle_subsets
        self.augmenters = []

    def __iter__(self):
        self.sampler_iterator = iter(self.sampler)
        return self

    def __next__(self):
        if self.shuffle_subsets:
            subset_names = self.get_subset_names()
            subset_idx = torch.randint(0, len(subset_names), (1,))
            self.select_subset(subset_names[subset_idx.item()])
        batch = self.dataset[next(self.sampler_iterator)]
        for augmenter in self.augmenters:
            augmenter.augment(batch)
        return batch

    def __len__(self):
        return self.n_batches

    def get_subset_names(self):
        if self.dataset.has_subsets:
            return self.dataset.get_subset_names()
        else:
            return ['DEFAULT']

    def select_subset(self, subset_name):
        if subset_name is not 'DEFAULT':
            self.dataset.select_subset(subset_name)

    def current_subset(self):
        return self.dataset.current_subset

    def set_augmenters(self, augmenters):
        self.augmenters = augmenters
