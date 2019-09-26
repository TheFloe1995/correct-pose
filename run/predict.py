"""
Run a model on a given datasets and write the predictions to a file.
"""

import os
import torch

from data_utils import datasets

# Config
dataset_name = 'HANDS17_V2V_test_poses'
model_path = 'results/hands17sc_mlp_finconf5_6/3/'
repetition_number = 0
device = torch.device('cuda:2')
output_name = 'HANDS17sc_V2V_mlp_5630_test'
########################################################################

dataset = datasets.SinglePoseDataset(dataset_name, device=device)

hyperparams = torch.load(os.path.join(model_path, 'params.pt'))
model = hyperparams['model'](hyperparams['model_args'])
model.to(device)

weights = torch.load(os.path.join(model_path, str(repetition_number), 'weights.pt'),
                     map_location=device)
model.load_state_dict(weights)

model.eval()
predictions = model(dataset.poses)

torch.save(predictions, os.path.join('results', 'predictions', output_name + '.pt'))
