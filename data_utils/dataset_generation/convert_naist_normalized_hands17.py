"""
Script to convert predictions and labels from NAIST DPREN network (HANDS 2017 challenge solution) to
the desired format. For the training of the DPREN network the data was transformed into pixel space
(uvd), then each sample was cropped and scaled individually. For the resulting predictions
(and labels) that means they have to be shifted, scaled and transformed back into camera space
(xyz). Parameters for the cropping/scaling are only available for the test set but not for the
training set. Therefore these parameters have to be restored by solving a simple linear system for
each sample by comparing the NAIST transformed labels and the original HANDS 2017 dataset labels.
Note that the NAIST version of the dataset is missing some samples (sorted out because of bad
quality).
"""

import os
import torch
import numpy as np

from data_utils.raw_data_handling import HANDS17Handler, NAISTNormalizedHANDS17Handler
from data_utils import helper

# Config
train_predictions_file_path = 'data/predictor_results/BigHand2.2M/DeepREN.npy'
test_predictions_file_path = 'data/predictor_results/BigHand2.2M/DeepREN_test_uvd.npy'
normalized_labels_dir = 'data/HANDS_2017_RVLab_normalized_labels'
original_labels_dir = 'data/'
output_path = 'data/'
output_name = 'HANDS17_DPREN_ShapeSplitPruned'
########################################################################

print("Loading and reformatting data.")
normalized_handler = NAISTNormalizedHANDS17Handler(normalized_labels_dir)
original_handler = HANDS17Handler(original_labels_dir)
train_predictions = torch.from_numpy(np.load(train_predictions_file_path).astype('float32'))
test_predictions = torch.from_numpy(np.load(test_predictions_file_path).astype('float32'))
normalized_labels = torch.from_numpy(normalized_handler.get_labels())

print('Removing missing samples from labels.')
normalized_label_ids = normalized_handler.get_frame_ids()
original_labels = torch.from_numpy(original_handler.get_labels(normalized_label_ids))

assert original_labels.shape == normalized_labels.shape

print('Transforming predictions back into original space.')
camera_params = {'fx': 475.065948,
                 'fy': 475.065857,
                 'cx': 315.944855,
                 'cy': 245.287079}
uvd_labels = helper.xyz_to_uvd(original_labels, camera_params)
scalings, shifts = helper.reconstruct_scaling_and_shift(normalized_labels, uvd_labels)
denormalized_train_predictions = helper.denorm_naist_poses(train_predictions, scalings, shifts)
xyz_train_predictions = helper.uvd_to_xyz(denormalized_train_predictions, camera_params)

scalings, shifts = helper.simplify_naist_denorm_params(normalized_handler.get_test_norm_params())
denormalized_test_predictions = helper.denorm_naist_poses(test_predictions, scalings, shifts)
xyz_test_predictions = helper.uvd_to_xyz(denormalized_test_predictions, camera_params)

print('Saving results.')
torch.save(xyz_train_predictions, os.path.join(output_path, '{}_all_poses.pt'.format(output_name)))
torch.save(original_labels, os.path.join(output_path, '{}_all_labels.pt'.format(output_name)))
torch.save(xyz_test_predictions, os.path.join(output_path, '{}_test_poses.pt'.format(output_name)))

print('Finished.')
