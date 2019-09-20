import torch
import torch.nn as nn
from collections import OrderedDict

from data_utils import pose_features


class Conv1x1LReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1LReLU, self).__init__()
        self.add_module('Conv1x1', nn.Conv1d(in_channels, out_channels, 1, 1))
        self.add_module('LeakyReLU', nn.LeakyReLU(0.1))


class FingerConvLReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(FingerConvLReLU, self).__init__()
        self.add_module('FingerConv', nn.Conv2d(in_channels, out_channels, (1, 5), 1))
        self.add_module('LeakyReLU', nn.LeakyReLU(0.05))


class MLP(nn.Sequential):
    def __init__(self, n_in, n_out, hidden_dims, activation_func, dropout=None, batchnorm=False):
        modules = OrderedDict()
        in_out_dims = [n_in] + hidden_dims
        for i, (in_dim, out_dim) in enumerate(zip(in_out_dims[:-1], in_out_dims[1:])):
            layer = OrderedDict({'linear': nn.Linear(in_dim, out_dim),
                                 'activation': activation_func})
            if batchnorm:
                layer['batchnorm'] = nn.BatchNorm1d(out_dim)
            modules['layer_' + str(i)] = nn.Sequential(layer)

        last_layer = OrderedDict({'linear': nn.Linear(hidden_dims[-1], n_out)})
        if dropout is not None:
            last_layer['dropout'] = nn.Dropout(dropout)
        last_layer['activation'] = activation_func
        modules['output_layer'] = nn.Sequential(last_layer)

        super(MLP, self).__init__(modules)


class MessageMLP(nn.Sequential):
    def __init__(self, n_nodes, latent_dim, layer_dims):
        super(MessageMLP, self).__init__()
        self.add_module('MLP', MLP(n_nodes * latent_dim, latent_dim, layer_dims, nn.ReLU()))


class MLPCoder(nn.Module):
    def __init__(self, in_channels, out_channels, layer_dims):
        super(MLPCoder, self).__init__()
        self.in_channels = in_channels
        self.out_cannels = out_channels
        self.net = MLP(21 * in_channels, 21 * out_channels, layer_dims, nn.LeakyReLU(0.05))

    def forward(self, x):
        result = self.net(x.reshape(-1, 21 * self.in_channels))
        return result.reshape(-1, 21, self.out_cannels)


# Encoder/decoder module that uses the same weights to encode/decode each of the joints or feature
# vectors it gets as input.
class Fully1x1ConvCoder(nn.Module):
    def __init__(self, in_channels, out_channels, layer_dims):
        super(Fully1x1ConvCoder, self).__init__()

        self.net = nn.Sequential()
        in_out_dims = [in_channels] + layer_dims + [out_channels]
        for i, (in_dim, out_dim) in enumerate(zip(in_out_dims[:-1], in_out_dims[1:])):
            self.net.add_module('Layer_' + str(i), Conv1x1LReLU(in_dim, out_dim))

    def forward(self, x):
        result = x.transpose(1, 2)
        result = self.net(result)
        return result.transpose(1, 2)


# Encoder/Decoder module that uses an own MLP for each of the joints or feature vectors it gets as
# input. This requires less parameters than using a fully connected network on all of them at once.
class StructuredMLPCoder(nn.Module):
    def __init__(self, in_channels, out_channels, layer_dims):
        super(StructuredMLPCoder, self).__init__()

        self.net = nn.ModuleList()
        for i in range(21):
            self.net.append(MLP(in_channels, out_channels, layer_dims, nn.LeakyReLU(0.1)))

    def forward(self, x):
        result = []
        for coder, node in zip(self.net, x.transpose(0, 1)):
            result.append(coder(node))

        return torch.stack(result, dim=1)


# Encoder/decoder model that uses own weights for each joint of a finger but shares weights between
# fingers.
class FullyFingerConvCoder(nn.Module):
    def __init__(self, in_channels, out_channels, layer_dims):
        super(FullyFingerConvCoder, self).__init__()

        self.net = nn.Sequential()
        in_out_dims = [in_channels] + layer_dims + [out_channels]
        for i, (in_dim, out_dim) in enumerate(zip(in_out_dims[:-1], in_out_dims[1:])):
            self.net.add_module('Layer_' + str(i), FingerConvLReLU(in_dim, out_dim))

    def forward(self, x):
        fingers = pose_features.joints_of_all_fingers(x).permute(0, 3, 1, 2)
        result = self.net(fingers)
        return result


# Standard message passing module for a graph with 21 joints following the hand skeleton topology.
# In addition to the regular bones, auxiliary edges are inserted between neighboring MCP joints.
# It calculates a message for each node based on its own feature vector and the feature vectors of
# all neighboring nodes. The feature vectors are concatenated and passed to externally defined
# message modules.
class MessagePassing(nn.Module):
    def __init__(self, latent_dim, message_module, message_module_dims):
        super(MessagePassing, self).__init__()

        self.nodes = nn.ModuleDict({
            # The wrist itself + 5 neighboring MCP joints --> 6
            'wrist': message_module(6, latent_dim, message_module_dims),
            # The remaining message modules are grouped by finger. First add just the modules for
            # the MCP joints, because they are a little irregular. Thumb and pinky have only 4
            # neighbors (Wrist + MCP + DIP + 1 neighboring MCP) while the others have 5 neighbors
            # (same but with 2 neighboring MCPs).
            'thumb': nn.ModuleList([message_module(4, latent_dim, message_module_dims)]),
            'index': nn.ModuleList([message_module(5, latent_dim, message_module_dims)]),
            'middle': nn.ModuleList([message_module(5, latent_dim, message_module_dims)]),
            'ring': nn.ModuleList([message_module(5, latent_dim, message_module_dims)]),
            'pinky': nn.ModuleList([message_module(4, latent_dim, message_module_dims)])
        })

        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        # Then it's the same for all fingers. DIP and PIP have 3 neighbors (including themselves)
        # and the TIP has only 2 neighbors.
        for f_key in self.finger_names:
            self.nodes[f_key].append(message_module(3, latent_dim, message_module_dims))
            self.nodes[f_key].append(message_module(3, latent_dim, message_module_dims))
            self.nodes[f_key].append(message_module(2, latent_dim, message_module_dims))

        # Auxiliary data structure for more convenient access: A dictionary that holds for each
        # finger a list of length 5 with the indices from range [0, 20] belonging to this finger in
        # order [W, MCP, DIP, PIP, TIP]. Note that the wrist is always redundantly contained.
        self.idx_map = {name: pose_features.finger_indices(i) for i, name in
                        enumerate(self.finger_names)}

        # Auxiliary data structure to easily query the neighbors of each joint: A dictionary that
        # holds for each finger a list of length 4 with lists of different lengths containing the
        # indices from range [0, 20] that belong to the neighbors of each joint. Additionally it
        # contains the neighbors of the wrist joint as well.
        self.neighbor_indices = self._generate_neighbor_indices()

    def forward(self, x):
        result = torch.zeros_like(x)

        result[:, 0] = self._calc_message(x, self.nodes['wrist'], self.neighbor_indices['wrist'])
        for finger_name in self.finger_names:
            for idx, neighbor_indices, message_module in zip(self.idx_map[finger_name][1:],
                                                             self.neighbor_indices[finger_name],
                                                             self.nodes[finger_name]):
                result[:, idx] = self._calc_message(x, message_module, neighbor_indices)

        return result

    def _generate_neighbor_indices(self):
        neighbor_indices = {name: [] for name in self.idx_map}

        # Wrist: remember that the joint itself counts to the neighbors.
        neighbor_indices['wrist'] = [0] + [finger_indices[1] for finger_indices in
                                           self.idx_map.values()]

        # MCP joints
        neighbor_indices['thumb'].append(self.idx_map['thumb'][:3] + [self.idx_map['index'][1]])
        neighbor_indices['index'].append(self.idx_map['index'][:3] + [self.idx_map['thumb'][1],
                                                                    self.idx_map['middle'][1]])
        neighbor_indices['middle'].append(self.idx_map['middle'][:3] + [self.idx_map['index'][1],
                                                                      self.idx_map['ring'][1]])
        neighbor_indices['ring'].append(self.idx_map['ring'][:3] + [self.idx_map['middle'][1],
                                                                  self.idx_map['pinky'][1]])
        neighbor_indices['pinky'].append(self.idx_map['pinky'][:3] + [self.idx_map['ring'][1]])

        for finger_name in self.idx_map.keys():
            neighbor_indices[finger_name].append(self.idx_map[finger_name][1:4])
            neighbor_indices[finger_name].append(self.idx_map[finger_name][2:5])
            neighbor_indices[finger_name].append(self.idx_map[finger_name][3:5])

        return neighbor_indices

    @staticmethod
    def _calc_message(x, message_module, indices):
        message_input = torch.cat([x[:, i] for i in indices], dim=1)
        return message_module(message_input)


class IdentMock(nn.Module):
    def forward(self, x):
        return x

    def test(self, x):
        return x
