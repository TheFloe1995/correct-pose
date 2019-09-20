import torch

from unit_tests import helpers
from evaluation.evaluator import Evaluator
from data_utils import datasets
from data_utils import distorters
from data_utils import normalization as norm


example_results = {
    'sub1': {
        'coord_diff': torch.tensor([0.0, 2.0]),
        'distance': torch.tensor([0.0, 2.0]),
        'bone_length': torch.tensor([0.0, 2.0]),
        'proportion': torch.tensor([0.0, 2.0]),
    },
    'sub2': {
        'coord_diff': torch.tensor([1.0, 3.0]),
        'distance': torch.tensor([1.0, 3.0]),
        'bone_length': torch.tensor([2.0, 4.0]),
        'proportion': torch.tensor([2.0, 4.0]),
    }
}


def test_to_batch():
    batch_size = 42
    poses = torch.rand(batch_size, 21, 3)
    labels = poses + torch.ones(batch_size, 21, 3)
    batch = datasets.PoseCorrectionBatch(poses, labels, poses, labels)

    true_results = {
        'coord_diff': torch.ones(batch_size),
        'distance': torch.sqrt(3.0 * torch.ones(batch_size)),
        'bone_length': torch.zeros(batch_size),
        'proportion': torch.zeros(batch_size),
    }

    results_norm = Evaluator.to_batch(batch)
    results_orig = Evaluator.to_batch(batch, space='original')

    for metric_name in Evaluator.metric_names:
        assert torch.allclose(results_norm[metric_name], true_results[metric_name], atol=1e-6)
        assert torch.allclose(results_orig[metric_name], true_results[metric_name], atol=1e-6)


def test_to_model():
    distorter = distorters.NoDistorter()
    model = helpers.DummyModel()
    dataset_no_subs = datasets.NormalizedPairedPoseDataset('unit_test/dummy42',
                                                           distorter, norm.NoNorm, False,
                                                           device='cuda:0')
    dataset_subs = datasets.NormalizedPairedPoseDataset('unit_test/ident42', distorter, norm.NoNorm,
                                                        True, device='cuda:0')
    data_loader_no_subs = datasets.DataLoader(dataset_no_subs, 6)
    data_loader_subs = datasets.DataLoader(dataset_subs, 6)

    batch_size = 42
    true_results = {
        'coord_diff': torch.zeros(batch_size, device='cuda:0'),
        'distance': torch.zeros(batch_size, device='cuda:0'),
        'bone_length': torch.zeros(batch_size, device='cuda:0'),
        'proportion': torch.zeros(batch_size, device='cuda:0'),
    }

    results_norm_no_subs = Evaluator.to_model(data_loader_no_subs, model)
    results_orig_no_subs = Evaluator.to_model(data_loader_no_subs, model, space='original')
    results_norm_subs = Evaluator.to_model(data_loader_subs, model)
    results_orig_subs = Evaluator.to_model(data_loader_subs, model, space='original')

    for metric_name in Evaluator.metric_names:
        for subset_name in ['sub1', 'sub2']:
            assert torch.allclose(results_norm_subs[subset_name][metric_name],
                                  true_results[metric_name], atol=1e-5)
            assert torch.allclose(results_orig_subs[subset_name][metric_name],
                                  true_results[metric_name], atol=1e-5)
        assert torch.allclose(results_norm_no_subs['DEFAULT'][metric_name],
                              true_results[metric_name], atol=1e-5)
        assert torch.allclose(results_orig_no_subs['DEFAULT'][metric_name],
                              true_results[metric_name], atol=1e-5)


def test_to_dataset():
    distorter = distorters.NoDistorter()
    dataset_no_subs = datasets.NormalizedPairedPoseDataset('unit_test/dummy42', distorter,
                                                           norm.NoNorm, False, device='cuda:0')
    dataset_subs = datasets.NormalizedPairedPoseDataset('unit_test/ident42', distorter, norm.NoNorm,
                                                        True, device='cuda:0')
    data_loader_no_subs = datasets.DataLoader(dataset_no_subs, 6)
    data_loader_subs = datasets.DataLoader(dataset_subs, 6)

    batch_size = 42
    true_results = {
        'coord_diff': torch.zeros(batch_size, device='cuda:0'),
        'distance': torch.zeros(batch_size, device='cuda:0'),
        'bone_length': torch.zeros(batch_size, device='cuda:0'),
        'proportion': torch.zeros(batch_size, device='cuda:0'),
    }

    results_norm_no_subs = Evaluator.to_dataset(data_loader_no_subs)
    results_orig_no_subs = Evaluator.to_dataset(data_loader_no_subs, space='original')
    results_norm_subs = Evaluator.to_dataset(data_loader_subs)
    results_orig_subs = Evaluator.to_dataset(data_loader_subs, space='original')

    for metric_name in Evaluator.metric_names:
        for subset_name in ['sub1', 'sub2']:
            assert torch.allclose(results_norm_subs[subset_name][metric_name],
                                  true_results[metric_name], atol=1e-5)
            assert torch.allclose(results_orig_subs[subset_name][metric_name],
                                  true_results[metric_name], atol=1e-5)
        assert torch.allclose(results_norm_no_subs['DEFAULT'][metric_name],
                              true_results[metric_name], atol=1e-5)
        assert torch.allclose(results_orig_no_subs['DEFAULT'][metric_name],
                              true_results[metric_name], atol=1e-5)


def test_means_per_metric():
    true_mean_results = {
        'sub1': {
            'coord_diff': torch.tensor(1.0),
            'distance': torch.tensor(1.0),
            'bone_length': torch.tensor(1.0),
            'proportion': torch.tensor(1.0),
        },
        'sub2': {
            'coord_diff': torch.tensor(2.0),
            'distance': torch.tensor(2.0),
            'bone_length': torch.tensor(3.0),
            'proportion': torch.tensor(3.0),
        }
    }

    mean_results = Evaluator.means_per_metric(example_results)

    for subset_name in mean_results.keys():
        for metric_name in Evaluator.metric_names:
            assert torch.allclose(mean_results[subset_name][metric_name],
                                  true_mean_results[subset_name][metric_name])


def test_means_over_subsets():
    true_mean_results = {
        'coord_diff': torch.tensor(1.5),
        'distance': torch.tensor(1.5),
        'bone_length': torch.tensor(2.0),
        'proportion': torch.tensor(2.0),
    }

    mean_results = Evaluator.means_over_subsets(example_results)
    for metric_name in Evaluator.metric_names:
        assert torch.allclose(mean_results[metric_name], true_mean_results[metric_name])


def test_the_mean_that_rules_them_all():
    the_true_one_mean = torch.tensor(1.75)

    the_one_and_only = Evaluator.the_mean_that_rules_them_all(example_results)

    torch.allclose(the_one_and_only, the_true_one_mean)
