import torch

from data_utils import datasets
from data_utils import distorters
from data_utils import normalization as norm


dataset_size = 42
distorter_config = {
        'stds': [3.0, 5.0, 10.0],
        'layer_probs': [0.7, 0.25, 0.05],
        'layer_radii': [0.0, 6.0, 7.0],
        'confusion_prob': 0.02
}


def test_single_pose_dataset():
    plain_size = 20
    dataset_plain = datasets.SinglePoseDataset('unit_test/dummy42_poses', num_samples=plain_size,
                                               device='cpu')
    dataset_dict = datasets.SinglePoseDataset('unit_test/dummy42_dict_poses', device='cuda:0')

    all = dataset_plain[:]
    single_sub1 = dataset_dict[0]
    dataset_dict.select_subset('sub2')
    single_sub2 = dataset_dict[1]

    assert type(all) is datasets.PoseCorrectionBatch
    assert len(dataset_plain) == plain_size
    assert len(dataset_dict) == dataset_size
    assert all.poses.shape == (plain_size, 21, 3)
    assert all.poses.dtype == torch.float32
    assert all.device == torch.device('cpu')
    assert dataset_dict.device == torch.device('cuda:0')
    assert single_sub1.device == torch.device('cuda:0')
    assert single_sub1.poses.shape == (1, 21, 3)
    assert not torch.allclose(single_sub1.poses, single_sub2.poses)
    assert dataset_dict.get_subset_names() == ['sub1', 'sub2']
    assert dataset_dict.has_subsets


def test_paired_pose_dataset():
    plain_size = 20
    some_size = 10
    distorter = distorters.SyntheticDistorter(distorter_config)

    dataset_plain = datasets.PairedPoseDataset('unit_test/dummy42', distorter, False, plain_size,
                                               'cpu')
    dataset_dict = datasets.PairedPoseDataset('unit_test/dummy42_dict', distorter, True,
                                              device='cuda:0')

    some = dataset_plain[:some_size]
    single_sub1 = dataset_dict[0]
    dataset_dict.select_subset('sub2')
    single_sub2 = dataset_dict[1]

    assert type(some) is datasets.PoseCorrectionBatch
    assert len(dataset_plain) == plain_size
    assert len(dataset_dict) == dataset_size
    assert some.poses.shape == (some_size, 21, 3)
    assert some.poses.dtype == torch.float32
    assert some.poses.is_same_size(some.labels)
    assert some.device == torch.device('cpu')
    assert dataset_dict.device == torch.device('cuda:0')
    assert single_sub1.device == torch.device('cuda:0')
    assert single_sub1.poses.shape == (1, 21, 3)
    assert not torch.allclose(single_sub1.poses, single_sub2.poses)
    assert not torch.allclose(single_sub1.poses, single_sub1.labels)
    assert dataset_dict.get_subset_names() == ['sub1', 'sub2']
    assert dataset_dict.has_subsets


def test_normalized_paired_pose_dataset():
    plain_size = 20
    some_size = 10
    distorter = distorters.SyntheticDistorter(distorter_config)

    dataset_plain = datasets.NormalizedPairedPoseDataset('unit_test/dummy42', distorter,
                                                         norm.NoNorm, False, plain_size, 'cpu')
    dataset_dict = datasets.NormalizedPairedPoseDataset('unit_test/dummy42_dict', distorter,
                                                        norm.NoNorm, True, device='cuda:0')

    some = dataset_plain[:some_size]
    single_sub1 = dataset_dict[0]
    dataset_dict.select_subset('sub2')
    single_sub2 = dataset_dict[1]

    assert type(some) is datasets.PoseCorrectionBatch
    assert len(dataset_plain) == plain_size
    assert len(dataset_dict) == dataset_size
    assert some.poses.shape == (some_size, 21, 3)
    assert some.poses.dtype == torch.float32
    assert some.poses.is_same_size(some.labels)
    assert some.poses.is_same_size(some.original_poses)
    assert some.poses.is_same_size(some.original_labels)
    assert some.normalization_params is not None
    assert some.device == torch.device('cpu')
    assert dataset_dict.device == torch.device('cuda:0')
    assert single_sub1.device == torch.device('cuda:0')
    assert single_sub1.poses.shape == (1, 21, 3)
    assert not torch.allclose(single_sub1.poses, single_sub2.poses)
    assert not torch.allclose(single_sub1.poses, single_sub1.labels)
    assert dataset_dict.get_subset_names() == ['sub1', 'sub2']
    assert dataset_dict.has_subsets


def test_data_loader():
    distorter = distorters.NoDistorter()
    dataset_plain = datasets.NormalizedPairedPoseDataset('unit_test/dummy42', distorter,
                                                         norm.NoNorm, True, dataset_size, 'cuda:0')
    dataset_dict = datasets.NormalizedPairedPoseDataset('unit_test/dummy42_dict', distorter,
                                                        norm.NoNorm, True, dataset_size, 'cuda:0')
    true_sum_sub1 = dataset_dict[:].poses.sum()
    dataset_dict.select_subset('sub2')
    true_sum_sub2 = dataset_dict[:].poses.sum()

    data_loader_plain = datasets.DataLoader(dataset_plain, 6)
    data_loader_dict = datasets.DataLoader(dataset_dict, 6)

    plain_batch = next(iter(data_loader_plain))
    subset_names_plain = data_loader_plain.get_subset_names()
    data_loader_plain.select_subset(subset_names_plain[0])

    all_batches = {}
    sum_of_subsets = {}
    for subset_name in data_loader_dict.get_subset_names():
        data_loader_dict.select_subset(subset_name)
        all_batches[subset_name] = list(data_loader_dict)
        sum_of_subsets[subset_name] = sum(batch.poses.sum() for batch in all_batches[subset_name])

    assert type(plain_batch) is datasets.PoseCorrectionBatch
    assert subset_names_plain == ['DEFAULT']
    assert list(all_batches.keys()) == ['sub1', 'sub2']
    assert len(all_batches['sub1']) == 7
    assert type(all_batches['sub1'][0]) == datasets.PoseCorrectionBatch
    assert all_batches['sub1'][0].labels.shape == (6, 21, 3)
    assert torch.allclose(sum_of_subsets['sub1'], true_sum_sub1)
    assert torch.allclose(sum_of_subsets['sub2'], true_sum_sub2)


def test_standard_distorter():
    config = {'stds': [3.0, 5.0, 10.0],
              'layer_probs': [0.7, 0.25, 0.05],
              'layer_radii': [0.0, 6.0, 7.0],
              'confusion_prob': 0.02}
    poses = torch.normal(torch.zeros(dataset_size, 21, 3), 3.0).cuda()

    distorter = distorters.SyntheticDistorter(config)
    distorted_poses = distorter(poses)

    assert poses.is_same_size(distorted_poses)
    assert not torch.allclose(poses, distorted_poses)


def test_knn_predefined_distorter():
    config = {
        'source_name': 'HANDS17_DPREN_SubjClust_train',
        'knn_name': 'HANDS17_DPREN_SubjClust_train_labels_noshift_16',
        'strength_alpha': -4.0,
        'strength_loc': 0.85,
        'strength_scale': 0.01,
        'max_k': 2,
        'device': 'cuda:0',
        'stds': [3.0, 5.0, 10.0],
        'layer_probs': [0.7, 0.25, 0.05],
        'layer_radii': [0.0, 6.0, 7.0],
        'confusion_prob': 0.02
    }
    distorter = distorters.KNNPredefinedDistorter(config)

    distort_dataset = datasets.PairedPoseDataset('HANDS17_DPREN_SubjClust_train', distorter, False,
                                                 100, 'cuda:0')
    no_distort_dataset = datasets.PairedPoseDataset('HANDS17_DPREN_SubjClust_train',
                                                    distorters.NoDistorter(), True, 100, 'cuda:0')

    distort_batch = distort_dataset[:]
    no_distort_batch = no_distort_dataset[:]

    torch.allclose(distort_batch.poses, no_distort_batch.poses)
    torch.equal(distort_batch.labels, no_distort_batch.labels)
