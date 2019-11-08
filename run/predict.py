"""
Run a model on a given datasets and write the predictions to a file.
"""

import os
import torch

from data_utils import datasets

# Config
dataset_name = 'HANDS17_DPREN_test_poses'
model_path = 'results/hands17sc_gnn_final_best_1/0'
repetition_number = 0
device = torch.device('cuda:0')
output_name = 'HANDS17sc_DPREN_gnn_fb0_test'
batch_size = int(1e5)
########################################################################

dataset = datasets.SinglePoseDataset(dataset_name, device=device)
data_loader = datasets.DataLoader(dataset, batch_size, shuffle=False)

hyperparams = torch.load(os.path.join(model_path, 'params.pt'))
model = hyperparams['model'](hyperparams['model_args'])
model.to(device)

weights = torch.load(os.path.join(model_path, str(repetition_number), 'weights.pt'),
                     map_location=device)
model.load_state_dict(weights)

model.eval()
prediction_list = []
for batch in data_loader:
    prediction_list.append(model(batch.poses).detach().cpu())
predictions = torch.cat(prediction_list, dim=0)

torch.save(predictions, os.path.join('results', 'predictions', output_name + '.pt'))
