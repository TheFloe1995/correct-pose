"""
DEPRECATION WARNING: Not tested/updated recently.
Converts joint annotations from the NYU matlab file format to a binary file, pickled with torch, while selecting 21 out
of the 36 annotated keypoints. (Not really working/equivalent).
"""

import sys
import torch
import scipy.io

file_name = sys.argv[1]
output_file_name = sys.argv[2]

# For 21 joints model
kp_selection = [29,
                28, 23, 17, 11, 5,
                27, 25, 24,
                21, 19, 18,
                15, 13, 12,
                9,   7,  6,
                3,   1,  0]

data = scipy.io.loadmat(file_name)
joints_xyz = data['joint_xyz'][0][:, kp_selection].astype('float32')

torch_data = torch.from_numpy(joints_xyz)
torch_data = torch_data.reshape(-1, 21, 3)

# NYU seems to contain left hands (in contrast to BigHand and MSRA15), therefore mirroring is applied for consistency.
torch_data[:, :, 2] *= -1.0

torch.save(torch_data, output_file_name)