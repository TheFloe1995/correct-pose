"""
Train some MLPs on HANDS 2017 dataset variants.
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
    'name': 'hands17sc_mlp_nonorm_nodist_refine_1',
    'train_set': 'HANDS17_DPREN_SubjClust_train',
    'val_set': 'HANDS17_DPREN_SubjClust_val',
    'train_set_size': None,
    'val_set_size': None,
    'use_preset': True,
    'normalizer': None,
    'target_device': torch.device('cuda:8'),
    'n_repetitions': 2,
    'init_weights_path': None,
    'deterministic_mode': True
}

solver_config = {
        'log_frequency': 3,
        'log_loss': True,
        'log_grad': False,
        'verbose': False,
        'show_plots': False,
        'num_epochs': 30,
        'batch_size': 1024,
        'interest_keys': [],
        'val_example_indices': [0],
        'val_example_subset': 'DEFAULT'
    }

base_hyperparams = {
    'model': AdditivePoseCorrectorMLP,
    'loss_function': losses.CombinedMetricLoss([errors.coordinate_difference,
                                                errors.bone_length_error],
                                               [0.9, 0.1],
                                               ['squared', 'squared']),
    'loss_space': 'default',
    'eval_space': 'default',
    'optimizer': torch.optim.Adam,
    'optimizer_args': {
        'betas': [0.9, 0.999],
        'eps': 1e-08,
        'lr': 5e-5,
        'weight_decay': 2e-4
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
    'distorter_args': {
        'source_name': 'HANDS17_DPREN_SubjClust_train',
        'knn_name': 'HANDS17_DPREN_SubjClust_train_labels_shift_16',
        'strength_alpha': -4.0,
        'strength_loc': 0.85,
        'strength_scale': 0.01,
        'max_k': 2,
        'device': experiment_config['target_device'],
        'stds': [3.0, 5.0, 10.0],
        'layer_probs': [0.7, 0.25, 0.05],
        'layer_radii': [0.0, 6.0, 7.0],
        'confusion_prob': 0.02
    },
    'augmenters': []
}

mlp_base_args = {
    'hidden_dims': [1024, 1024, 1024],
    'activation_func': nn.LeakyReLU(0.1),
    'batchnorm': False,
    'dropout': 0.2
}

base_hyperparams['model_args'] = mlp_base_args

########################################################################

# Specify options for a hyperparameter grid search

#learning_rates = [5e-4, 7e-4]
#weight_decays = [1e-4, 2e-4]
#hidden_dims = [[1400, 1400], [1024, 1024, 1024]]
dropouts = [0.1, 0.2, 0.3]
#batchnorms = [True, False]

# distorter_args = [
#     {
#         'source_name': 'MSRA15_3DCNN_all_nonorm',
#         'strength_mean': 0.1,
#         'strength_std': 0.05,
#         'device': experiment_config['target_device']
#     },
#     {
#         'source_name': 'MSRA15_3DCNN_all_nonorm',
#         'strength_mean': 0.2,
#         'strength_std': 0.1,
#         'device': experiment_config['target_device']
#     },
#     {
#         'source_name': 'MSRA15_3DCNN_all_norm',
#         'strength_mean': 0.2,
#         'strength_std': 0.1,
#         'device': experiment_config['target_device']
#     },
#     {
#         'source_name': 'MSRA15_3DCNN_all_norm',
#         'strength_mean': 0.4,
#         'strength_std': 0.2,
#         'device': experiment_config['target_device']
#     }
# ]

#loss_spaces = ['default', 'original']
#activation_funcs = [nn.LeakyReLU(0.1), nn.ReLU()]
#loss_functions = [losses.CombinedMetricLoss([errors.coordinate_difference,
#                                             errors.bone_length_error],
#                                            [0.9, 0.1],
#                                            ['squared', 'squared']),
#                  losses.CombinedMetricLoss([errors.coordinate_difference,
#                                             errors.bone_length_error],
#                                               [0.8, 0.2],
#                                               ['squared', 'squared']),
#                  ]


#distorterss = [distorters.SyntheticDistorter, distorters.PredefinedDistorter]
#dist_source_names = ['MSRA15_3DCNN_all_nonorm', 'HANDS17_DPREN_ShapeSplit_train_nonorm']
#dist_strength_locs = [0.85, 1.0]
#dist_strength_scales = [0.01, 0.1]
#dist_max_k = [2, 3, 4]
# augmenters_list = [[augmenters.RandomInterpolator(-8.0, 1.1, 0.4)],
#                    [augmenters.RandomInterpolator(-16.0, 1.0, 0.3)]]

####################################################################################################

#################
# DO NOT FORGET #
#################

# Register the desired options to the experiment

experiment = Experiment(experiment_config, solver_config, base_hyperparams)

#experiment.add_options(('model_args', 'hidden_dims'), hidden_dims)
#experiment.add_options(('model_args', 'activation_func'), activation_funcs)
#experiment.add_options(('optimizer_args', 'lr'), learning_rates)
#experiment.add_options(('optimizer_args', 'weight_decay'), weight_decays)
#experiment.add_options((None, 'loss_function'), loss_functions)
#experiment.add_options((None, 'distorter_args'), distorter_args)
#experiment.add_options((None, 'distorter'), distorterss)
#experiment.add_options((None, 'loss_space'), loss_spaces)
experiment.add_options(('model_args', 'dropout'), dropouts)
#experiment.add_options(('model_args', 'batchnorm'), batchnorms)
#experiment.add_options(('distorter_args', 'source_name'), dist_source_names)
#experiment.add_options(('distorter_args', 'strength_loc'), dist_strength_locs)
#experiment.add_options(('distorter_args', 'strength_scale'), dist_strength_scales)
#experiment.add_options(('distorter_args', 'max_k'), dist_max_k)
#experiment.add_options((None, 'augmenters'), augmenters_list)

experiment.run()
