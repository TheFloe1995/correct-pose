import torch

from evaluation import metrics
from evaluation import errors
from data_utils import helper
from networks import modules


class Evaluator:
    metric_names = ['coord_diff', 'distance', 'bone_length', 'proportion']
    colors = {
        'black': '\033[00m',
        'green': '\033[92m',
        'red': '\033[91m'
    }

    @classmethod
    def to_batch(cls, batch, space='default', mode='mean'):
        results = {
            'coord_diff': cls._apply_metric(errors.coordinate_difference, batch, space, mode),
            'distance': cls._apply_metric(errors.distance_error, batch, space, mode),
            'bone_length': cls._apply_metric(errors.bone_length_error, batch, space, mode),
            'proportion': cls._apply_metric(errors.proportion_error, batch, space, mode),
        }
        return results

    @classmethod
    def to_model(cls, data_loader, model, space='default', mode='mean'):
        eval_results = {}
        for subset_name in data_loader.get_subset_names():
            data_loader.select_subset(subset_name)

            results = {metric_key: [] for metric_key in cls.metric_names}
            for batch in data_loader:
                batch.poses = model.test(batch.poses)

                if space == 'original':
                    normalizer = data_loader.dataset.normalizer
                    batch.original_poses = normalizer.denormalize(batch.poses,
                                                                  batch.normalization_params)

                batch_results = cls.to_batch(batch, space, mode)

                for metric_key, values in batch_results.items():
                    results[metric_key].append(values)

            eval_results[subset_name] = helper.map_func_to_dict(results, torch.cat)

        return eval_results

    @classmethod
    def to_dataset(cls, data_loader, space='default', mode='mean'):
        return cls.to_model(data_loader, modules.IdentMock(), space, mode)

    # Deprecation warning: Not used/updated/tested recently.
    # Computes how much the improvement on each metric declines after denormalization.
    # In other words: compares the improvements achieved in normalized and in denormalized space.
    @classmethod
    def denormalization_decline(cls, data_loader, model):
        norm_results_before = cls.to_dataset(data_loader, space='default')
        orig_results_before = cls.to_dataset(data_loader, space='original')

        norm_results_after = cls.to_model(data_loader, model, space='default')
        orig_results_after = cls.to_model(data_loader, model, space='original')

        decline = {}
        for metric_name in cls.metric_names:
            norm_rel_improvement = norm_results_after[metric_name] / norm_results_before[metric_name]
            orig_rel_improvement = orig_results_after[metric_name] / orig_results_before[metric_name]
            decline[metric_name] = orig_rel_improvement - norm_rel_improvement

        return decline

    @classmethod
    def print_comparison(cls, results, baseline_results):
        for eval_space in results.keys():
            print('\n{}:'.format(eval_space))
            for subset_name in results[eval_space].keys():
                print('\n\t{}:'.format(subset_name))
                print('\t\t\t\terrors\t  baseline   difference   relative difference')
                for metric_name in cls.metric_names:
                    value = results[eval_space][subset_name][metric_name]
                    baseline_value = baseline_results[eval_space][subset_name][metric_name]
                    difference = value - baseline_value
                    relative_improvement = difference / baseline_value

                    s = '\t\t{:<12}{:>10}{:.4f}{:>10}{:.4f}{:>10}{:.4f}{:>15.2%}{}'
                    color = cls._color_code_sign(difference)
                    s = s.format(metric_name, color, value, color, baseline_value, color,
                                 difference, relative_improvement, cls.colors['black'])
                    print(s)
            print()

    @classmethod
    def means_per_metric(cls, results):
        means = {}
        for subset_name, subset_results in results.items():
            means[subset_name] = helper.map_func_to_dict(subset_results, torch.mean)
        return means

    @classmethod
    def means_over_subsets(cls, results):
        means = {}
        for metric_name in cls.metric_names:
            all_results_on_single_metric = [subset_results[metric_name] for subset_results in
                                            results.values()]
            means[metric_name] = torch.stack(all_results_on_single_metric).mean()
        return means

    @classmethod
    def the_mean_that_rules_them_all(cls, results):
        return torch.stack(list(cls.means_over_subsets(results).values())).mean()

    @classmethod
    def print_results(cls, results, indent=0):
        indent_str = '\t' * indent
        for eval_space, eval_space_results in results.items():
            print('{}{}:'.format(indent_str, eval_space))
            for subset_name, subset_results in eval_space_results.items():
                print('\t{}{}'.format(indent_str, subset_name))
                for metric_name, val in subset_results.items():
                    print('\t\t{}{:<11}:{:>10.4f}'.format(indent_str, metric_name, val))

    @classmethod
    def print_result_summary_flat(cls, results, prefix=''):
        mean_results = cls.means_over_subsets(results)
        combined_mean = cls.the_mean_that_rules_them_all(results)
        s = '{}combi: {:.2e}'.format(prefix, combined_mean)
        for metric_name, metric_result in mean_results.items():
            s += '\t{}: {:.2e}'.format(metric_name, metric_result)
        print(s)

    @classmethod
    def results_to_cpu(cls, results):
        for subset_name, subset_results in results.items():
            for metric_name in cls.metric_names:
                subset_results[metric_name] = subset_results[metric_name].cpu()

    @classmethod
    def _apply_metric(cls, error_function, batch, space, mode='mean'):
        if space == 'default':
            poses = batch.poses
            labels = batch.labels
        else:
            poses = batch.original_poses
            labels = batch.original_labels

        if mode == 'mean':
            metric_func = metrics.mean_error
        elif mode == 'max':
            metric_func = metrics.max_error
        else:
            raise ValueError('Unknown mode: {}'.format(mode))

        return metric_func(error_function(poses, labels), mode='absolute').detach()

    @classmethod
    def _color_code_sign(cls, val):
        if val < -1e-3:
            color = cls.colors['green']
        elif val > 1e-3:
            color = cls.colors['red']
        else:
            color = cls.colors['black']
        return color
