import torch
import torch.nn as nn

from networks.pose_correctors import AdditivePoseCorrectorMLP, PoseCorrectorGNNv1
from networks import modules

torch.autograd.set_detect_anomaly(True)


def test_additive_pose_corrector_mlp():
    model_input = torch.autograd.Variable(torch.rand(42, 21, 3, device='cuda'))

    models = [AdditivePoseCorrectorMLP({'hidden_dims': [10], 'activation_func': nn.ReLU(),
                                        'dropout': 0.5, 'batchnorm': True}),
              AdditivePoseCorrectorMLP({'hidden_dims': [10, 10], 'activation_func': nn.ReLU(),
                                        'dropout': None, 'batchnorm': False})]

    for model in models:
        model.cuda()
        model_output = model(model_input)
        model_output.sum().backward()

        assert model_input.is_same_size(model_output)


def test_pose_corrector_gnn():
    config_1 = {
        'n_iter': 2,
        'latent_dim': 10,
        'encoder': modules.MLPCoder,
        'encoder_dims': [3, 5],
        'message_passing': modules.MessagePassing,
        'message_module': modules.MessageMLP,
        'message_passing_dims': [10, 10],
        'decoder': modules.Fully1x1ConvCoder,
        'decoder_dims': [6, 7],
        'dropout': 0.0
    }

    config_2 = {
        'n_iter': 2,
        'latent_dim': 10,
        'encoder': modules.StructuredMLPCoder,
        'encoder_dims': [3, 5],
        'message_passing': modules.MessagePassing,
        'message_module': modules.MessageMLP,
        'message_passing_dims': [20, 20],
        'decoder': modules.StructuredMLPCoder,
        'decoder_dims': [16, 8],
        'dropout': 0.1
    }

    models = [PoseCorrectorGNNv1(config_1), PoseCorrectorGNNv1(config_2)]

    for model in models:
        model_input = torch.autograd.Variable(torch.rand(42, 21, 3, device='cuda'))
        model.cuda()
        model_output = model(model_input)
        model_output.sum().backward()

        assert model_input.is_same_size(model_output)
