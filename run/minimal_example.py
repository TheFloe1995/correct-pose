"""
Minimal example training script for testing a fresh installation.
Can be run as it is after specifying the target device
"""

import torch
import torch.nn as nn

from networks.pose_correctors import *
from networks import modules
from training import losses
from training.experiment import Experiment
from evaluation import errors
from data_utils import distorters
from data_utils import normalization as norm
from data_utils import augmenters


# Base configuration

experiment_config = {
    'name': 'temp_minimal_example',
    'train_set': 'HANDS17_DPREN_SubjClust_train',
    'val_set': 'HANDS17_DPREN_SubjClust_val',
    'train_set_size': None,
    'val_set_size': None,
    'use_preset': True,
    'normalizer': None,
    'target_device': torch.device('cuda:X'),  # Replace X by a number or use 'cpu'
    'n_repetitions': 2,
    'init_weights_path': None
}

solver_config = {
        'log_frequency': 3,
        'log_loss': True,
        'log_grad': True,
        'verbose': False,
        'show_plots': False,
        'num_epochs': 10,
        'batch_size': 512,
        'interest_keys': [],
        'val_example_indices': [0],
        'val_example_subset': 'DEFAULT'
    }

base_hyperparams = {
    'model': AdditivePoseCorrectorMLP,
    'loss_function': losses.SingleMetricLoss(errors.distance_error, 'squared'),
    'loss_space': 'default',
    'eval_space': 'default',
    'optimizer': torch.optim.Adam,
    'optimizer_args': {
        'betas': [0.9, 0.999],
        'eps': 1e-08,
        'lr': 5e-4,
        'weight_decay': 1e-4
    },
    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'scheduler_requires_metric': True,
    'scheduler_args': {
        'mode': 'min',
        'factor': 0.5,
        'patience': 8,
        'verbose': True
    },
    'distorter': distorters.NoDistorter,
    'distorter_args': None,
    'augmenters': []
}

mlp_base_args = {
    'hidden_dims': [256, 256],
    'activation_func': nn.LeakyReLU(0.1),
    'batchnorm': False,
    'dropout': 0.2
}

base_hyperparams['model_args'] = mlp_base_args

########################################################################

# Specify options for a hyperparameter grid search

learning_rates = [5e-4, 7e-4]
weight_decays = [1e-4, 2e-4]

####################################################################################################

#################
# DO NOT FORGET #
#################

# Register the desired options to the experiment

experiment = Experiment(experiment_config, solver_config, base_hyperparams)

experiment.add_options(('optimizer_args', 'lr'), learning_rates)
experiment.add_options(('optimizer_args', 'weight_decay'), weight_decays)

experiment.run()
