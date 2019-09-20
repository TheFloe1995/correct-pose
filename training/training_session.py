import torch
import os
import time

from evaluation.evaluator import Evaluator
from data_utils import helper


class TrainingException(Exception):
    def __init__(self, loss):
        self.loss = loss


class TrainingSession:
    """
    Class for a single training session for a specific model with fixed hyper-parameters.
    The class encapsulates low level training logic associated with the model and optimizer i.e.
    running the model's forward/backward path, updating model parameters, run model in test mode and
    scheduling the learning rate.
    """
    def __init__(self, model, hyper_params, normalizer):
        if not type(hyper_params) is dict:
            raise Exception("Error: The passed config object is not a dictionary.")
        self.params = hyper_params
        self.model = model
        self.device = model.device
        self.normalizer = normalizer

        self.loss_function = hyper_params['loss_function']
        self.optimizer = hyper_params['optimizer'](model.parameters(),
                                                   **(hyper_params['optimizer_args']))

        self.scheduler = hyper_params['scheduler'](self.optimizer,
                                                   **(hyper_params['scheduler_args']))
        if self.params["scheduler_requires_metric"]:
            self.scheduler_metric = float('inf')

    def test_model(self, data_loader):
        results = Evaluator.to_model(data_loader, self.model, self.params['eval_space'])
        return Evaluator.means_per_metric(results)

    def schedule_learning_rate(self):
        if self.params['scheduler_requires_metric']:
            self.scheduler.step(self.scheduler_metric)
        else:
            self.scheduler.step()

    def train_batch(self, batch):
        helper.check_device_compatibility(batch.list_tensors(), self.device)
        self.optimizer.zero_grad()

        poses = torch.autograd.Variable(batch.poses)

        net_output = self.model(poses)
        loss = self._backward_and_optimize(net_output, batch, self.normalizer)
        if type(net_output) is tuple:
            batch.poses = net_output[0]
        else:
            batch.poses = net_output

        return loss, batch

    def _backward_and_optimize(self, outputs, input_batch, normalizer):
        if self.params['loss_space'] == 'original':
            if type(outputs) is tuple:
                poses = normalizer.denormalize(outputs[0], input_batch.normalization_params)
                outputs = (poses, *outputs[1:])
            else:
                outputs = normalizer.denormalize(outputs, input_batch.normalization_params)

            loss = self.loss_function(outputs, input_batch.original_labels)
        else:
            loss = self.loss_function(outputs, input_batch.labels)

        loss.backward()
        self.optimizer.step()
        loss = loss.data.cpu()
        return loss

    @staticmethod
    def _dump_debug_info(loss, input_poses, predictions, labels, weights):
        debug_dir = 'DEBUG'
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)

        session_dir = os.path.join(debug_dir, str(time.time()))
        os.mkdir(session_dir)

        torch.save(loss, os.path.join(session_dir, 'loss.pt'))
        torch.save(input_poses, os.path.join(session_dir, 'input.pt'))
        torch.save(predictions, os.path.join(session_dir, 'pred.pt'))
        torch.save(labels, os.path.join(session_dir, 'labels.pt'))
        torch.save(weights, os.path.join(session_dir, 'weights.pt'))
