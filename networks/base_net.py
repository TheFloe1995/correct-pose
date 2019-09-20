from abc import ABC, abstractmethod
import torch.nn as nn


class BaseNet(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def test(self, data):
        self.eval()
        results = self.forward(data)
        self.train()
        return results

    @property
    def device(self):
        """
        Return the device the model is stored on.
        It is assumed that all weights are always only residing on a single device.
        """
        return next(self.parameters()).device

    def num_parameters(self):
        return sum([t.nelement() for _, t in self.state_dict().items()])
