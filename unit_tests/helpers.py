import torch.nn as nn


class DummyLoss(nn.Module):
    def forward(self, predictions, labels):
        return predictions.sum()


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net = nn.Linear(63, 63)

    def forward(self, pose_batch):
        return self.net(pose_batch.reshape(-1, 63)).reshape(-1, 21, 3)

    def test(self, pose_batch):
        return pose_batch

    @property
    def device(self):
        return next(self.parameters()).device
