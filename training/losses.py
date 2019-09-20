import torch.nn as nn

from evaluation import metrics


# Combine different metrics/errors in a weighted sum.
# Parameters:
#   error_funcs: list of function names from evaluation.errors
#   weights: list of floats of the same length as error_funcs, specifying the weights for the sum
#   metric_modes: list of strings of the same length as error_funcs, specifying the mode of the
#                 metric that should be used (usually either 'squared' or 'absolute')
class CombinedMetricLoss(nn.Module):
    def __init__(self, error_funcs, weights, metric_modes):
        super(CombinedMetricLoss, self).__init__()
        self.error_funcs = error_funcs
        self.weights = weights
        self.metric_modes = metric_modes

    def forward(self, poses, labels):
        loss = 0.0
        for error_func, weight, mode in zip(self.error_funcs, self.weights, self.metric_modes):
            loss += weight * metrics.mean_error(error_func(poses, labels), mode).mean()
        return loss

    def __str__(self):
        return 'Error functions:{} \nWeights: {}\tMetric modes: {}'.format(self.error_funcs,
                                                                           self.weights,
                                                                           self.metric_modes)

    def __eq__(self, other):
        return str(self) == str(other)


# Using just a single metric is just a special case of the combined setting.
class SingleMetricLoss(CombinedMetricLoss):
    def __init__(self, error_func, metric_mode):
        super(SingleMetricLoss, self).__init__([error_func], [1.0], [metric_mode])
