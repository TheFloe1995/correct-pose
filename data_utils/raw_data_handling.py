# Collection of classes for loading the raw data without any preprocessing from the original file
# formats and directory structures of the original datasets.

import os
import glob
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseHandler(ABC):
    def __init__(self, base_dir):
        self.base_dir = base_dir

    # Handlers are expected to return the labels as a numpy array of type float32 and with shape
    # (n_samples, n_joints, n_dimensions).
    @abstractmethod
    def get_labels(self):
        pass

    @property
    @abstractmethod
    def n_samples(self):
        pass


# For loading data from the original HANDS 2017 challenge dataset as provided by the authors.
class HANDS17Handler(BaseHandler):
    def __init__(self, base_dir):
        super().__init__(base_dir)

        label_file_path = os.path.join(base_dir, 'HANDS17_labels.txt')
        self.data = pd.read_csv(label_file_path, sep='\t', usecols=range(64), header=None)
        self.data.set_index(pd.Index(self.get_frame_ids(), verify_integrity=True), inplace=True)

    def get_labels(self, ids=slice(None)):
        return self.data.loc[ids][range(1, 64)].values.astype('float32').reshape(-1, 21, 3)

    @property
    def n_samples(self):
        return len(self.data)

    def get_frame_ids(self):
        image_file_names = self.data[0]
        frame_ids = [file_name[7:15] for file_name in image_file_names]
        return frame_ids


# For loading data from the altered format used by NAIST RV-Lab for their solution to the challenge.
class NAISTNormalizedHANDS17Handler(BaseHandler):
    def __init__(self, base_dir):
        super().__init__(base_dir)

        # Load labels for the training set. They are stored as packs of approximately 1000 labels
        # in compressed pickle files.
        train_label_data = []
        train_label_packs_paths = sorted(glob.glob(os.path.join(self.base_dir, '*.pkl')))
        for i, pack_path in enumerate(train_label_packs_paths):
            if i % 100 == 0:
                print('Loading pack {}/{}'.format(i, len(train_label_packs_paths)))
            pack_data = pd.read_pickle(pack_path, compression='gzip')
            train_label_data.append(pack_data)
        self.train_label_data = pd.concat(train_label_data)

        # Load the normalization params for the test set from similar packs.
        test_norm_params = []
        test_norm_params_dir = os.path.join(self.base_dir, 'test_norm_params')
        test_norm_param_packs_paths = sorted(glob.glob(os.path.join(test_norm_params_dir, '*.pkl')))
        for pack_path in test_norm_param_packs_paths:
            pack_data = pd.read_pickle(pack_path, compression='gzip')
            test_norm_params.append(pack_data)
        self.test_norm_params = pd.concat(test_norm_params)

    def get_labels(self):
        return np.stack(self.train_label_data.values).astype('float32').reshape(-1, 21, 3)

    def get_test_norm_params(self):
        return self.test_norm_params.values

    @property
    def n_samples(self):
        return len(self.train_label_data)

    def get_frame_ids(self):
        naist_ids = self.train_label_data.index
        frame_ids = [naist_id[9:] for naist_id in naist_ids]
        return frame_ids
