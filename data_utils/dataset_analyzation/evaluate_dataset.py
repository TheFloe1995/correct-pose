"""
Compute the mean evaluation metrics on a dataset (before applying any corrections).
"""

import os
import torch

from data_utils import datasets
from evaluation.evaluator import Evaluator

# Config
dataset_name = 'HANDS17_DPREN_SubjClust_val'
########################################################################

dataset = datasets.PairedPoseDataset(dataset_name, use_preset=True)
data_loader = datasets.DataLoader(dataset, 100000)
results = Evaluator.means_per_metric(Evaluator.to_dataset(data_loader, 'default'))

torch.save(results, os.path.join('results', 'datasets', dataset_name + '.pt'))
