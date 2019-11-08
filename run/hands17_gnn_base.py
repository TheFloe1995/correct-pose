"""
Train some GNNs on HANDS 2017 dataset variants.
"""

import torch

from networks.pose_correctors import PoseCorrectorGNNv1
from networks import modules
from training import losses
from training.experiment import Experiment
from evaluation import errors
from data_utils import distorters
from data_utils import normalization as norm


# Base configuration

experiment_config = {
    'name': 'hands17sc_gnn_finconf11_1',
    'train_set': 'HANDS17_DPREN_SubjClust_small5_train',
    'val_set': 'HANDS17_DPREN_SubjClust_val',
    'train_set_size': None,
    'val_set_size': None,
    'use_preset': False,
    'normalizer': None,
    'target_device': torch.device('cuda:2'),
    'n_repetitions': 1,
    'init_weights_path': None,
    'deterministic_mode': True
}

solver_config = {
        'log_frequency': 10,
        'log_loss': True,
        'log_grad': False,
        'verbose': False,
        'show_plots': False,
        'num_epochs': 80,
        'batch_size': 2048,
        'interest_keys': [],
        'val_example_indices': [0],
        'val_example_subset': 'DEFAULT'
    }

base_hyperparams = {
    'model': PoseCorrectorGNNv1,
    'loss_function': losses.CombinedMetricLoss([errors.coordinate_difference,
                                                errors.bone_length_error],
                                               [0.8, 0.2],
                                               ['absolute', 'absolute']),
    'loss_space': 'default',
    'eval_space': 'default',
    'optimizer': torch.optim.Adam,
    'optimizer_args': {
        'betas': [0.9, 0.999],
        'eps': 1e-08,
        'lr': 1e-4,
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
    'distorter': distorters.GaussianDistorter,
    'distorter_args': {
        'source_name': 'HANDS17_DPREN_SubjClust_small5_train',
        'knn_name': 'HANDS17_DPREN_SubjClust_small5_train_labels_shift_8',
        'strength_alpha': -4.0,
        'strength_loc': 0.94,
        'strength_scale': 0.02,
        'max_k': 4,
        'device': experiment_config['target_device'],
        'normal_stats_name': 'normal_params'
    },
    'augmenters': []
}

gnn_base_args = {
    'n_iter': 8,
    'latent_dim': 24,
    'encoder': modules.Fully1x1ConvCoder,
    'encoder_dims': [32, 32],
    'message_passing': modules.MessagePassing,
    'message_module': modules.MessageMLP,
    'message_passing_dims': [64, 64],
    'decoder': modules.Fully1x1ConvCoder,
    'decoder_dims': [32, 32],
    'rand_iter': False,
    'dropout': 0.0
}

base_hyperparams['model_args'] = gnn_base_args

########################################################################

# Specify options for a hyperparameter grid search

#learning_rates = [5e-4, 1e-3]
#weight_decays = [1e-4, 5e-5]
#n_iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#loss_spaces = ['original', 'default']
#distort_strengths_stds = [0.1, 0.2]
#latent_dims = [24, 32, 40]
#message_passing_dims = [[64, 64], [128]]
#norm_activ_funcs = [nn.LeakyReLU(0.1), nn.LeakyReLU(0.01)]

loss_functions = [losses.CombinedMetricLoss([errors.coordinate_difference,
                                             errors.bone_length_error],
                                            [0.8, 0.2],
                                            ['squared', 'squared']),
                  losses.CombinedMetricLoss([errors.coordinate_difference,
                                             errors.bone_length_error],
                                            [0.9, 0.1],
                                            ['absolute', 'absolute']),
                  ]

model_args_list = [{
        'n_iter': 4,
        'latent_dim': 24,
        'encoder': modules.MLPCoder,
        'encoder_dims': [32, 32],
        'message_passing': modules.MessagePassing,
        'message_module': modules.MessageMLP,
        'message_passing_dims': [64, 64],
        'decoder': modules.MLPCoder,
        'decoder_dims': [32, 32],
        'rand_iter': False,
        'dropout': 0.0
    },
    {
        'n_iter': 4,
        'latent_dim': 24,
        'encoder': modules.MLPCoder,
        'encoder_dims': [64],
        'message_passing': modules.MessagePassing,
        'message_module': modules.MessageMLP,
        'message_passing_dims': [64, 64],
        'decoder': modules.MLPCoder,
        'decoder_dims': [64],
        'rand_iter': False,
        'dropout': 0.0
    }
]

#distorterss = [distorters.NoDistorter, distorters.PredefinedDistorter]
#dist_max_k = [2, 3]
#dist_strength_means = [0.9]
#dist_strength_stds = [0.01, 0.05]
#rand_iters = [True, False]
dropouts = [0.05, 0.1, 0.2]
####################################################################################################

#################
# DO NOT FORGET #
#################

# Register the desired options to the experiment

experiment = Experiment(experiment_config, solver_config, base_hyperparams)

#experiment.add_options(('optimizer_args', 'lr'), learning_rates)
#experiment.add_options(('optimizer_args', 'weight_decay'), weight_decays)
#experiment.add_options((None, 'model_args'), model_args)
#experiment.add_options(('model_args', 'n_iter'), n_iters)
#experiment.add_options(('model_args', 'latent_dim'), latent_dims)
#experiment.add_options(('model_args', 'dropout'), dropouts)
#experiment.add_options(('model_args', 'message_passing_dims'), message_passing_dims)
#experiment.add_options((None, 'loss_function'), loss_functions)
#experiment.add_options((None, 'distorter_args'), distorter_args)
#experiment.add_options((None, 'distorter'), distorterss)
#experiment.add_options((None, 'loss_space'), loss_spaces)
#experiment.add_options(('distorter_args', 'strengths_std'), distort_strengths_stds)
#experiment.add_options(('distorter_args', 'max_k'), dist_max_k)
#experiment.add_options(('distorter_args', 'strength_mean'), dist_strength_means)

experiment.run()
