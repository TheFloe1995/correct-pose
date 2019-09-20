import os
import torch
import matplotlib.pyplot as plt
import pprint
import numpy as np

from evaluation.evaluator import Evaluator
from data_utils import datasets
from evaluation import metrics


# After executing an experiment this class provides functionality to analyze the results easy and
# fast. It loads the results and logs of all models trained during the experiment and extracts
# information about the best performing ones and which parameters were best.
class ExperimentAnalyzer:
    def __init__(self, experiment_name):
        self.experiment_dir = os.path.join('results', experiment_name)
        self.config = torch.load(os.path.join(self.experiment_dir, 'config.pt'))

        self.hyperparams = []
        self.log = []
        self.results = []

        self.string_repr_param_names = [(None, 'loss_function'), (None, 'augmenters')]

        # Each subdir corresponds to a single hyperparameter setting (session), for which
        # n_repetitions models have been trained.
        sub_dir_names = next(os.walk(self.experiment_dir))[1]
        for sub_dir_name in sorted(sub_dir_names, key=lambda x: int(x)):
            session_dir = os.path.join(self.experiment_dir, sub_dir_name)
            self.hyperparams.append(torch.load(os.path.join(session_dir, 'params.pt'),
                                               map_location='cpu'))

            self.log.append([])
            self.results.append([])
            for i in range(self.config['experiment']['n_repetitions']):
                repetition_dir = os.path.join(session_dir, str(i))

                self.log[-1].append(torch.load(os.path.join(repetition_dir, 'log.pt'),
                                               map_location='cpu'))
                eval_results = torch.load(os.path.join(repetition_dir, 'eval.pt'),
                                          map_location='cpu')

                # In a special experimental setting, the GNN is trained with varying numbers of
                # internal iterations k. At the end of training, results for a range of values for k
                # are stored in a dictionary and saved. If this is the case, the results dictionary
                # has an extra layer and the 'default' key is not present. Therefore the best
                # performing k has to be found first.
                if 'default' in eval_results:
                    self.results[-1].append(eval_results)
                else:
                    best_k = self._find_best_k(eval_results, self.hyperparams[-1]['eval_space'])
                    self.results[-1].append(eval_results[best_k])
                    self.log[-1][i]['best_k'] = best_k

        # During the experiment a grid search over hyperparameters of interest has been conducted.
        # Each hyperparameter of interest received multiple values in different sessions. The
        # following auxiliary data structured allow to query which parameters settings performed
        # best on which metric.

        # This table lists for each value of each parameter the indices of the models that have been
        # trained with this value.
        self.param_value_index_mapping = {}
        self._compute_param_value_index_mapping()

        # This table just reformats the results such that the results of all models and repetitions
        # for a single metric can be accessed via a single tensor.
        self.per_metric_results = {}
        self._aggregate_results_per_metric()

        # This table reformats the results such that all the results belonging to a single parameter
        # value can be accessed easily via a single tensor.
        self.per_param_value_results = {}
        self._aggregate_results_per_param_value()

    def print_best_model_summary(self):
        for eval_space, eval_space_results in self.per_metric_results.items():
            subset_names = self.results[0][0][eval_space].keys()
            n_metrics = eval_space_results.shape[0]
            n_sessions = eval_space_results.shape[1]
            n_repetitions = eval_space_results.shape[2]

            best_session_indices = set()

            print('{}:'.format(eval_space))
            for i, subset_name in enumerate(subset_names):
                print('\n\t{}:'.format(subset_name))
                min_repetition_vals, repetition_indices = torch.min(eval_space_results[..., i],
                                                                    dim=2)
                min_vals, session_indices = torch.min(min_repetition_vals, dim=1)
                best_session_indices.update(set(session_indices.tolist()))
                variances = torch.var(eval_space_results.reshape(n_metrics, -1), dim=1,
                                      unbiased=False)

                print('\t\t\t\tValue\tIndices\tVariance')
                for j, metric_name in enumerate(Evaluator.metric_names):
                    s = '\t\t{:<11}{:>12.4f}\t({}, {})\t{:.2e}'
                    s_idx = session_indices[j]
                    r_idx = repetition_indices[j, s_idx]
                    s = s.format(metric_name, min_vals[j], s_idx, r_idx, variances[j])
                    print(s)

            print('\n\tBest on average:')
            reshaped_results = eval_space_results.reshape(n_metrics, n_sessions, n_repetitions, -1)
            mean_results = reshaped_results.mean(dim=3)
            min_repetition_vals, repetition_indices = torch.min(mean_results, dim=2)
            min_vals, session_indices = torch.min(min_repetition_vals, dim=1)
            best_session_indices.update(set(session_indices.tolist()))

            print('\t\t\t\tValue\tIndices')
            for i, metric_name in enumerate(Evaluator.metric_names):
                s = '\t\t{:<11}{:>12.4f}\t({}, {})'
                s_idx = session_indices[i]
                r_idx = repetition_indices[i, s_idx]
                s = s.format(metric_name, min_vals[i], s_idx, r_idx)
                print(s)

            print('\n\tHyperparameters:')
            print('\t\tParams of interest: {}'.format(self.config['solver']['interest_keys']))
            for i in best_session_indices:
                print('\t\tSession {}: \t{}'.format(i, self._format_interest_val_list(i)))

    def plot_average_model_performances(self, metric_name):
        fig, ax = plt.subplots(1, 2, figsize=(15.0, 10.0))
        for i, (eval_space, eval_space_results) in enumerate(self.per_metric_results.items()):
            metric_idx = Evaluator.metric_names.index(metric_name)
            n_sessions = eval_space_results.shape[1]
            mean_results = eval_space_results[metric_idx].reshape(n_sessions, -1).mean(dim=1)

            self._bar_plot(mean_results, ax[i], eval_space)

        plt.show()

    def print_params_ranking(self):
        for eval_space, eval_space_results in self.per_param_value_results.items():
            print('{}:'.format(eval_space))
            for param_name, param_results in eval_space_results.items():
                print('\n\t{}:'.format(param_name))
                n_values = param_results.shape[0]
                n_metrics = param_results.shape[1]
                mean_results = param_results.reshape(n_values, n_metrics, -1).mean(dim=2)
                min_results, indices = torch.min(mean_results, dim=0)
                variances = torch.var(mean_results, dim=0)

                print('\t\t\t\tAverage result\tVariance\tParam value')
                for i, metric_name in enumerate(Evaluator.metric_names):
                    best_value = self.param_value_index_mapping[param_name][0][indices[i]]
                    print('\t\t{:<11}\t{:4f}\t{:.2e}\t{}'.format(metric_name, min_results[i],
                                                                 variances[i], best_value))

    def plot_average_parameter_performances(self, param_name, metric_name):
        fig, ax = plt.subplots(1, 2, figsize=(15.0, 5.0))
        for i, (eval_space, eval_space_results) in enumerate(self.per_param_value_results.items()):
            metric_idx = Evaluator.metric_names.index(metric_name)
            n_values = eval_space_results[param_name].shape[0]
            reshaped_results = eval_space_results[param_name][:, metric_idx].reshape(n_values, -1)
            mean_results = reshaped_results.mean(dim=1)

            self._bar_plot(mean_results, ax[i], eval_space)

        plt.show()

    def print_info(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config['experiment'])
        print()
        pp.pprint(self.config['solver'])
        print()
        pp.pprint(self.hyperparams[0])
        print(self.hyperparams[0]['loss_function'])

    def _compute_param_value_index_mapping(self):
        for param_name in self.config['solver']['interest_keys']:
            values = []
            indices = []
            for i, session_params in enumerate(self.hyperparams):
                if param_name[0] is None:
                    val = session_params[param_name[1]]
                else:
                    val = session_params[param_name[0]][param_name[1]]

                if param_name in self.string_repr_param_names:
                    if type(val) is list:
                        val = ''.join(str(o) for o in val)
                    else:
                        val = str(val)

                if val not in values:
                    values.append(val)
                    indices.append([i])
                else:
                    indices[values.index(val)].append(i)
            self.param_value_index_mapping[param_name] = (values, indices)

    def _aggregate_results_per_metric(self):
        for eval_space in self.results[0][0].keys():
            subset_names = self.results[0][0][eval_space].keys()
            aggregation_matrix = torch.zeros(len(Evaluator.metric_names),
                                             len(self.results),
                                             self.config['experiment']['n_repetitions'],
                                             len(subset_names))

            for i, metric_name in enumerate(Evaluator.metric_names):
                for j, session_results in enumerate(self.results):
                    for k, repetition_results in enumerate(session_results):
                        for l, subset_name in enumerate(subset_names):
                            aggregation_matrix[i, j, k, l] = \
                            repetition_results[eval_space][subset_name][metric_name]

            self.per_metric_results[eval_space] = aggregation_matrix

    def _aggregate_results_per_param_value(self):
        for eval_space, eval_space_results in self.per_metric_results.items():
            self.per_param_value_results[eval_space] = {}
            for param_name, value_index_mapping in self.param_value_index_mapping.items():
                param_values = value_index_mapping[0]
                indices = value_index_mapping[1]
                aggregation_matrix = torch.zeros(len(param_values),
                                                 eval_space_results.shape[0],
                                                 len(indices[0]),
                                                 eval_space_results.shape[2],
                                                 eval_space_results.shape[3])
                for i in range(len(param_values)):
                    aggregation_matrix[i] = eval_space_results[:, indices[i]]
                self.per_param_value_results[eval_space][param_name] = aggregation_matrix

    def _format_interest_val_list(self, i):
        interest_keys = self.config['solver']['interest_keys']
        s = ''
        for k1, k2 in interest_keys:
            if k1 is None:
                val = self.hyperparams[i][k2]
            else:
                val = self.hyperparams[i][k1][k2]
            s += '{},  '.format(val)
        return s

    @staticmethod
    def _bar_plot(values, ax, title):
        max_val = values.max()
        ax.set_xlim(0.0, 1.2 * max_val)

        ax.barh(range(len(values)), values)
        ax.set_title(title)
        for i, val in enumerate(values):
            ax.text(1.05 * max_val, i, '{:.4f}'.format(val.item()), color='black',
                    fontweight='bold')

    @classmethod
    def _find_best_k(cls, eval_results, eval_space):
        lowest_distance_error = float('inf')
        best_k = -1
        for k, results in eval_results.items():
            distance_error = results[eval_space]['DEFAULT']['distance']
            if distance_error < lowest_distance_error:
                lowest_distance_error = distance_error
                best_k = k
        return best_k


# This class provides further functionality to analyze and compare the performance of a single
# model.
class ModelAnalyzer:
    def __init__(self, model_directory, hyperparams, dataset, training_config, batch_size=100):
        self.dataset = dataset
        self.data_loader = datasets.DataLoader(self.dataset, batch_size, shuffle=False)

        self.training_config = training_config
        self.hyperparams = hyperparams
        self.model = hyperparams['model'](hyperparams['model_args'])
        weights = torch.load(os.path.join(model_directory, 'weights.pt'), map_location='cpu')
        try:
            self.model.load_state_dict(weights)
        except RuntimeError:
            print('WARNING: Unable to load state dict')
        self.model.cuda()

        self.log = torch.load(os.path.join(model_directory, 'log.pt'), map_location='cpu')
        self.logged_results = torch.load(os.path.join(model_directory, 'eval.pt'),
                                         map_location='cpu')

        self.errors = None
        self.dataset_errors = None
        self.predictions = None

    def evaluate_model(self, mode='mean'):
        self.errors = {'default': Evaluator.to_model(self.data_loader, self.model, space='default',
                                                     mode=mode)}
        if type(self.dataset) == datasets.NormalizedPairedPoseDataset:
            self.errors['original'] = Evaluator.to_model(self.data_loader, self.model,
                                                         space='original', mode=mode)

    def evaluate_dataset(self, precompute_prefix=None, mode='mean'):
        if precompute_prefix is not None:
            path = self._construct_error_file_path(precompute_prefix)
            self.dataset_errors = torch.load(path)
        else:
            self.dataset_errors = {'default': Evaluator.to_dataset(self.data_loader,
                                                                   space='default', mode=mode)}
            if type(self.dataset) == datasets.NormalizedPairedPoseDataset:
                self.dataset_errors['original'] = Evaluator.to_dataset(self.data_loader,
                                                                       space='original', mode=mode)

    def test_model(self, result_space='default'):
        predictions = []
        for batch in self.data_loader:
            pred = self.model.test(batch.poses)
            if result_space == 'original':
                pred = self.data_loader.dataset.normalizer.denormalize(pred,
                                                                       batch.normalization_params)
            predictions.append(pred.detach().cpu())
        self.predictions = torch.cat(predictions)

    def compare_success_rate_to(self, other_errors, space, subset_name, metric_name):
        those_errors = self.errors[space][subset_name][metric_name]
        other_errors = other_errors[space][subset_name][metric_name]
        max_error = max(torch.max(those_errors), torch.max(other_errors))
        thresholds = np.linspace(0.0, max_error.item(), 100)

        this_curve = metrics.success_rate_curve(those_errors.cpu().numpy(), thresholds)
        other_curve = metrics.success_rate_curve(other_errors.cpu().numpy(), thresholds)

        self._plot_success_rate_curves([this_curve, other_curve], thresholds, ['this', 'other'])

    def plot_logs(self, percentage=1.0):
        fig, axes = plt.subplots(2, 3, sharex='col', figsize=(3 * 5, 2 * 5))

        train_log = self.log['train']
        train_keys = set(train_log.keys())
        val_log = self.log['val']

        log_length = int(percentage * len(next(iter(val_log.values()))[Evaluator.metric_names[0]]))

        dataset_size = self.training_config['experiment']['train_set_size']
        batch_size = self.training_config['solver']['batch_size']
        log_frequency = self.training_config['solver']['log_frequency']
        log_iterations = torch.arange(1, log_length + 1)
        log_iterations *= dataset_size // batch_size // log_frequency
        log_iterations = log_iterations.tolist()

        if 'loss' in train_keys:
            loss_log_length = int(percentage * len(train_log['loss']))
            axes[0, 2].plot(train_log['loss'][-loss_log_length:])
            axes[0, 2].set_title('loss')
            train_keys.remove('loss')

        if 'grad' in train_keys:
            grad_log_length = int(percentage * len(train_log['grad']))
            axes[1, 2].plot(train_log['grad'][-grad_log_length:])
            axes[1, 2].set_title('grad')
            train_keys.remove('grad')

        for i, metric_name in enumerate(Evaluator.metric_names):
            row_idx = i // 2
            col_idx = i % 2
            ax = axes[row_idx, col_idx]

            train_values = train_log[metric_name][-log_length:]
            if len(train_values) > 0:
                ax.plot(log_iterations, train_values, label='train')

            for subset_name, subset_values in val_log.items():
                val_values = subset_values[metric_name][-log_length:]
                ax.plot(log_iterations, val_values, label='val-{}'.format(subset_name))

            ax.legend()
            ax.set_title(metric_name)

        plt.show()

    def compare_results(self):
        model_results = {}
        dataset_results = {}
        for eval_space in self.errors.keys():
            model_results[eval_space] = Evaluator.means_per_metric(self.errors[eval_space])
            dataset_results[eval_space] = Evaluator.means_per_metric(self.dataset_errors[eval_space])
        Evaluator.print_comparison(model_results, dataset_results)

    def save_dataset_errors(self, prefix):
        self._save_errors(self.dataset_errors, prefix)

    def save_model_errors(self, prefix):
        self._save_errors(self.errors, prefix)

    def print_hyperparameters(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.hyperparams)

    @staticmethod
    def _plot_success_rate_curves(curves, thresholds, names):
        for curve, name in zip(curves, names):
            plt.plot(thresholds, curve, label=name)
        plt.legend()
        plt.xlabel('thresholds')
        plt.ylabel('success rate in %')

    @classmethod
    def _save_errors(cls, errors, prefix):
        path = cls._construct_error_file_path(prefix)
        torch.save(errors, path)

    @staticmethod
    def _construct_error_file_path(prefix):
        return os.path.join('results', '{}_errors.pt'.format(prefix))
