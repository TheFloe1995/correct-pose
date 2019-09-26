from abc import ABC, abstractmethod

from networks.base_net import BaseNet
from networks import modules


# The correction is computed by a simple fully connected neural network. Optionally batch
# normalization can be added after every layer or dropout after the last layer.
# Additive means that the internal network predicts only correction vectors for each joint which is
# then added to the input pose at the end instead of predicting the corrected pose directly. If the
# pose is already quite good it simply needs to learn how to predict zeros.
# Model always expects a 21x3 pose as input (and returns 21x3 corrected pose as well).
# Config:
#   hidden_dims: list of ints, specifyint the number of neurons per hidden layer
#   activation_func: nn.Module which is used as activation function in the whole network
#   dropout: float or None, specifies dropout probability in the last layer (or no dropout at all)
#   batchnorm: bool, whether to use batch normalization in every layer or not at all
class AdditivePoseCorrectorMLP(BaseNet):
    def __init__(self, config):
        super(AdditivePoseCorrectorMLP, self).__init__()
        self.net = modules.MLP(63, 63, config['hidden_dims'], config['activation_func'],
                              config['dropout'], config['batchnorm'])

    def forward(self, x):
        x = x + self.net(x.reshape(x.shape[0], -1)).reshape(*x.shape)
        return x


# This model is a fully connected network just like the AdditivePoseCorrectorMLP and works in the
# same additive manner. But instead of predicting a correction for each joint individually, it
# computes a single shift vector that is applied to all joints in the same way.
class GlobalShiftCorrectorMLP(BaseNet):
    def __init__(self, config):
        super(GlobalShiftCorrectorMLP, self).__init__()
        self.net = modules.MLP(63, 3, config['hidden_dims'], config['activation_func'])

    def forward(self, x):
        x = x + self.net(x.reshape(x.shape[0], -1)).view(-1, 1, 3)
        return x


# GNNs for pose corretions always consist of 3 stages.
#   1. Encoder: Takes the input pose and encodes it into a set of feature vectors for the nodes of
#               graph the model is working on.
#   2. Message Passing: Over n_iter iterations, the message passing module defines how each feature
#                       vector is updated based on its neighbors (and own state).
#   3. Decoder: Takes all feature vectors as input and decodes them back into the pose space.
# The architecture of the 3 modules can be specified externally. Details of the forward pass and how
# the modules are connected to each other have to be defined in the derived classes.
# Config:
#   n_iter: number of internal message passing iterations.
#   latent_dim: length of each feature vector
#   encoder: class (type name) of the encoder module
#   encoder_dims: list of ints, specifying the size of the encoder layers
#   message_passing: class (type name) of the message passing module
#   message_module: class (type name) of the message module
#   message_passing_dims: list of ints, specifying the size of the encoder layers
#   decoder: class (type name) of the decoder module
#   decoder_dims: list of ints, specifying the size of the decoder layers
class BasePoseCorrectorGNN(BaseNet, ABC):
    def __init__(self, config):
        super(BasePoseCorrectorGNN, self).__init__()

        self.n_iter = config['n_iter']

        self.encoder = config['encoder'](3, config['latent_dim'], config['encoder_dims'])
        self.message_passing = config['message_passing'](config['latent_dim'],
                                                         config['message_module'],
                                                         config['message_passing_dims'],
                                                         config['dropout'])
        self.decoder = config['decoder'](config['latent_dim'], 3, config['decoder_dims'])

    @abstractmethod
    def forward(self, x):
        pass


# Residual connections are applied inside the unrolled recurrent message passing block. That means,
# the message passing layers are computing an additive update for each state. An additional residual
# connection is applied directly from the input pose to the output pose. That means, the network
# predicts an additive correction for the joint positions instead of directly estimating the
# corrected pose.
class PoseCorrectorGNNv1(BasePoseCorrectorGNN):
    def forward(self, x):
        node_features = self.encoder(x)

        for k in range(self.n_iter):
            node_features = node_features + self.message_passing(node_features)

        x = x + self.decoder(node_features)
        return x
