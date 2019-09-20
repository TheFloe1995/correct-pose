import torch

from unit_tests import helpers
from training.training_session import TrainingSession
from training.solver import Solver
from training.losses import SingleMetricLoss
from data_utils import datasets
from data_utils import distorters
from data_utils import normalization as norm
from evaluation.evaluator import Evaluator
from evaluation import errors


model = helpers.DummyModel()
model.cuda()
hyperparams = {
    'loss_function': helpers.DummyLoss(),
    'loss_space': 'original',
    'eval_space': 'original',
    'optimizer': torch.optim.Adam,
    'optimizer_args': {
        'betas': [0.9, 0.999],
        'eps': 1e-08,
        'lr': 1e-3,
        'weight_decay': 0.0
    },
    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'scheduler_requires_metric': True,
    'scheduler_args': {
        'mode': 'min',
        'factor': 0.75,
        'patience': 2,
        'verbose': True
    },
    'augmenters': []
}

distorter_args = {
    'stds': [3.0, 5.0, 10.0],
    'layer_probs': torch.tensor([0.7, 0.25, 0.05]),
    'layer_radii': [0.0, 6.0, 7.0],
    'confusion_prob': 0.02
}
distorter = distorters.SyntheticDistorter(distorter_args)


def test_training_session():
    training_set = datasets.NormalizedPairedPoseDataset('unit_test/dummy42', distorter, norm.NoNorm,
                                                        False, None, device='cuda:0')
    validation_set = datasets.NormalizedPairedPoseDataset('unit_test/dummy42_dict',
                                                          distorters.NoDistorter(), norm.NoNorm,
                                                          True, None, device='cuda:0')
    training_batch = training_set[:]
    val_loader = datasets.DataLoader(validation_set, batch_size=6)

    training_session = TrainingSession(model, hyperparams, norm.NoNorm)
    training_session.schedule_learning_rate()
    loss, result = training_session.train_batch(training_batch)
    test_results = training_session.test_model(val_loader)

    assert loss.numel() == 1
    assert loss.device == torch.device('cpu')
    assert training_batch.poses.is_same_size(result.poses)
    assert list(test_results.keys()) == ['sub1', 'sub2']
    assert list(test_results['sub1'].keys()) == Evaluator.metric_names
    assert test_results['sub1']['distance'].numel() == 1


def test_solver():
    training_set = datasets.NormalizedPairedPoseDataset('unit_test/dummy42', distorter, norm.NoNorm,
                                                        False, None, device='cuda:0')
    validation_set = datasets.NormalizedPairedPoseDataset('unit_test/dummy42_dict',
                                                          distorters.NoDistorter(), norm.NoNorm,
                                                          True, None, device='cuda:0')

    solver_params = {
        'log_frequency': 2,
        'log_loss': True,
        'log_grad': True,
        'verbose': False,
        'show_plots': False,
        'num_epochs': 5,
        'batch_size': 6,
        'interest_keys': [(None, 'loss_function'),
                          ('optimizer_args', 'weight_decay'),
                          ('optimizer_args', 'lr')],
        'val_example_indices': [0, 1],
        'val_example_subset': 'sub1'
    }

    solver = Solver(solver_params, training_set, validation_set)
    log, eval_results, weights, example_predictions = solver.train(model, hyperparams)

    assert len(log['train']['loss']) == 35
    assert len(log['val']['sub1'][Evaluator.metric_names[0]]) == 10
    assert example_predictions.shape == (10, 2, 21, 3)
    assert type(eval_results) is dict
    assert list(eval_results.keys()) == ['default', 'original']
    assert log['train']['loss'][0].device == torch.device('cpu')
    assert log['train']['distance'][0].device == torch.device('cpu')
    assert log['val']['sub1']['distance'][0].device == torch.device('cpu')


def test_single_metric_loss():
    labels = torch.zeros(42, 21, 3, device='cuda')
    poses = torch.ones(42, 21, 3, device='cuda')
    metric_mode = 'absolute'
    loss_func = SingleMetricLoss(errors.coordinate_difference, metric_mode)

    true_loss = torch.tensor(1.0, device='cuda')

    loss = loss_func(poses, labels)

    assert true_loss.is_same_size(loss)
    assert torch.allclose(loss, true_loss)
