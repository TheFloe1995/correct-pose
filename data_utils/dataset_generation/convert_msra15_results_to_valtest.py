"""
DEPRECATION WARNING: Not tested/updated recently.
Script to convert a given list of predictor result text files for MSRA15 to a single binary file, while also changing
the joint order.
"""

import sys
import os
import numpy as np
import torch

# Params for validation set
#   data/MSRA15_val_poses.pt
#   data/predictor_results/MSRA/REN_9x6x6.txt
#   data/predictor_results/MSRA/3DCNN.txt
#   data/predictor_results/MSRA/HandPointNet.txt
# Params for test set
#   data/MSRA15_test_poses.pt
#   data/predictor_results/MSRA/Pose_REN.txt
#   data/predictor_results/MSRA/SHPR_Net.txt
#   data/predictor_results/MSRA/Point_to_Point.txt
# MSRA camera parameters
fx, fy, ux, uy = 241.42, 241.42, 160, 120

output_file_name = sys.argv[1]
input_file_names = sys.argv[2:]

joint_order = [0, 17, 1, 5, 9, 13, 18, 19, 20, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
subject_split_indices = [0, 8499, 16991, 25403, 33891, 42391, 50888, 59385, 67883, 76375]
test_subject = 8

results = {}
for file_name in input_file_names:
    data = np.loadtxt(file_name).astype('float32')
    start_idx = subject_split_indices[test_subject]
    end_idx = subject_split_indices[test_subject + 1]
    data = data[start_idx:end_idx]
    data = data.reshape(-1, 21, 3)

    # The predictions are stored in uvd coordinates and need to be transformed to xyz
    data[:, :, 0] = (data[:, :, 0] - ux) * data[:, :, 2] / fx
    data[:, :, 1] = (data[:, :, 1] - uy) * data[:, :, 2] / fy

    # For some reason the predicted coordinates are rotated by 180 degrees around the z axis compared to the original
    # labels from the dataset website. Turn them back.
    data[:, :, 1:] *= -1.0

    data = data[:, joint_order]
    data = torch.from_numpy(data)
    results[os.path.splitext(os.path.split(file_name)[-1])[0]] = data

torch.save(results, output_file_name)