import torch
from datetime import datetime as dt
import copy

from data_utils.visualization import PoseVisualizer
from evaluation.evaluator import Evaluator
from training.training_session import TrainingSession
from data_utils import datasets
from data_utils import helper


class Solver(object):
    """
    Solver class for to train models. A single instance can be used to train multiple models. The
    class encapsulates the high level training logic, e.g. loading data, starting training sessions,
    logging, deciding when to stop training...
    """
    def __init__(self, config, train_set, val_set):
        """
        Initialize a new Solver object.
        :param config: Dictionary with parameters defining the general solver behavior, e.g.
                       verbosity, logging, etc.
        :type config: dict

        :param train_set: Training data set that returns tuples of 2 pose tensors when iterated.
        :type train_set: torch.utils.data.DataSet

        :param val_set: Validation data set that returns tuples of 2 pose tensors when iterated.
        :type train_set: torch.utils.data.DataSet
        """
        if not type(config) is dict:
            raise Exception("Error: The passed config object is not a dictionary.")
        self.config = config

        if train_set.use_preset and train_set.has_subsets:
            shuffle_subsets = True
        else:
            shuffle_subsets = False
        self.train_loader = datasets.DataLoader(train_set, config['batch_size'], shuffle=True,
                                                shuffle_subsets=shuffle_subsets)
        self.val_loader = datasets.DataLoader(val_set, min(len(val_set), 10000), shuffle=False)
        if hasattr(self.train_loader.dataset, 'normalizer'):
            self.normalizer = self.train_loader.dataset.normalizer
        else:
            self.normalizer = None

        self.iters_per_epoch = len(self.train_loader)

    def train(self, model, hyperparams):
        """
        Trains the passed model on the training set with the specified hyper-parameters.
        Loss, validation errors or other intermediate results are logged (or printed/plotted during
        the training) and returned at the end, together with the best weights according to the
        validation performance.

        :param model: Model to train.
        :type model: torch.nn.Module

        :param hyperparams: Dictionary with all the hyperparameters required for training.
        :type hyperparams: dict

        :return: log: Dictionary containing all logs collected during training.
        :type: log: dict

        :return: final_val_results: The validation results (all metrics) of the model using the best
                                    weights after training finished.
        :type: final_val_results: dict

        :return: best_weights: Model weights that performed best on the validation set during the
                               whole training.
        :type: best_weights: dict

        :return: example_predictions: Corrected example poses from the validation set collected
                                      during training.
        :type: example_predictions: torch.FloatTensor
        """
        start_time = dt.now()
        print('Time: ', start_time.strftime('%H:%M:%S'))
        print()

        print('Setting things up...')

        session = TrainingSession(model, hyperparams, self.normalizer)
        self.train_loader.set_augmenters(hyperparams['augmenters'])

        helper.print_hyperparameters(hyperparams, self.config['interest_keys'], indent=1)
        log, example_predictions, log_iterations = self._initialize_logs()

        print('\n\tChecking initial validation performance:')
        best_val_performance, best_weights = self._initial_performance(session)

        print()
        print('All set, let\'s get started!')
        for epoch in range(self.config['num_epochs']):
            running_loss = 0.0
            session.schedule_learning_rate()
            for i, batch in enumerate(self.train_loader):
                loss, output_batch = session.train_batch(batch)

                if self.config['log_loss']:
                    log['train']['loss'].append(loss)

                if self.config['log_grad']:
                    log['train']['grad'].append(self._sum_gradients(model))

                if i in log_iterations:
                    train_performance, val_performance = self._intermediate_eval(session,
                                                                                 output_batch)
                    self._logging(log, loss, train_performance, val_performance, i)
                    if len(self.config['val_example_indices']) > 0:
                        example_predictions.append(self._example_predictions(session))

                running_loss += loss

            # Training for this epoch finished.
            session.scheduler_metric = running_loss / self.iters_per_epoch

            val_performance = session.test_model(self.val_loader)
            target_performance = Evaluator.means_over_subsets(val_performance)['distance']
            if target_performance < best_val_performance:
                best_val_performance = target_performance
                best_weights = copy.deepcopy(model.state_dict())
                log['best_epoch'] = epoch

            self._print_epoch_end_info(session, val_performance, start_time, epoch,
                                       best_val_performance)

        model.load_state_dict(best_weights)
        final_val_results = self._full_evaluation(model, session.params['eval_space'])
        print()
        print('-' * 30)
        print('FINISH')
        print('Final validation errors:')
        Evaluator.print_results(final_val_results)
        print()

        return log, final_val_results, best_weights, torch.stack(example_predictions).cpu()

    def _initial_performance(self, session):
        val_start = dt.now()
        validation_performance = session.test_model(self.val_loader)
        target_performance = Evaluator.means_over_subsets(validation_performance)['distance']
        val_time = dt.now() - val_start

        Evaluator.print_result_summary_flat(validation_performance, '\t')
        print('\t\tThat took {} milliseconds.'.format(val_time.microseconds // 1000))

        return target_performance, copy.deepcopy(session.model.state_dict())

    def _initialize_logs(self):
        log = {'train': {metric_name: [] for metric_name in Evaluator.metric_names},
               'val': {subset_name: {key: [] for key in Evaluator.metric_names}
                       for subset_name in self.val_loader.get_subset_names()}}
        log['train']['loss'] = []
        log['train']['grad'] = []
        example_predictions = []
        log_iterations = self._get_log_iterations(self.config['log_frequency'])
        return log, example_predictions, log_iterations

    def _get_log_iterations(self, log_frequency):
        logging_epochs = [(self.iters_per_epoch - 1) - log * (self.iters_per_epoch // log_frequency)
                          for log in range(0, log_frequency)]
        return logging_epochs

    def _intermediate_eval(self, session, batch):
        if session.params['eval_space'] == 'original':
            batch.original_poses = self.normalizer.denormalize(batch.poses,
                                                               batch.normalization_params)
        train_results = {'DEFAULT': Evaluator.to_batch(batch, session.params['eval_space'])}
        train_mean_results = Evaluator.means_per_metric(train_results)
        Evaluator.results_to_cpu(train_mean_results)

        val_mean_results = session.test_model(self.val_loader)
        Evaluator.results_to_cpu(val_mean_results)

        return train_mean_results, val_mean_results

    def _logging(self, log, loss, train_performance, val_performance, i):
        for key, value in train_performance['DEFAULT'].items():
            log['train'][key].append(value)

        for subset_key, subset_performance in val_performance.items():
            for key, value in subset_performance.items():
                log['val'][subset_key][key].append(value)

        if self.config['verbose']:
            print('Iteration {}/{}:'.format(i, self.iters_per_epoch - 1, loss))
            print('Loss: {:.2e}'.format(loss))
            Evaluator.print_result_summary_flat(train_performance, '\tTraining  : ')
            Evaluator.print_result_summary_flat(val_performance, '\tValidation: ')
            print()

    def _example_predictions(self, session):
        current_subset = self.val_loader.current_subset()
        self.val_loader.select_subset(self.config['val_example_subset'])
        example_batch = self.val_loader.dataset[self.config['val_example_indices']]
        self.val_loader.select_subset(current_subset)

        example_predictions = session.model.test(example_batch.poses).detach()

        if self.config['show_plots']:
            PoseVisualizer.triplet(example_batch.poses, example_batch.labels, example_predictions)

        return example_predictions

    def _full_evaluation(self, model, eval_space):
        default_results = Evaluator.means_per_metric(Evaluator.to_model(self.val_loader, model,
                                                                        'default'))
        eval_results = {'default': default_results}

        if eval_space == 'original':
            original_results = Evaluator.means_per_metric(Evaluator.to_model(self.val_loader, model,
                                                                             'original'))
            eval_results['original'] = original_results
        for eval_space_results in eval_results.values():
            Evaluator.results_to_cpu(eval_space_results)

        return eval_results

    def _print_epoch_end_info(self, session, val_performance, start_time, epoch,
                              best_val_performance):
        val_distance_error = val_performance[self.config['val_example_subset']]['distance']

        loss_performance_str = 'Average loss: {:.3e}, Validation performance: {:.3e}'.format(
            session.scheduler_metric, val_distance_error)
        display_str = self._buffered_str(loss_performance_str)
        print(display_str)

        epoch_time_str = 'Epoch: {}/{},\tTime: {},\tBest validation performance: {}'.format(
            epoch + 1, self.config['num_epochs'], self._get_elapsed_time_str(start_time),
            best_val_performance)
        display_str = self._buffered_str(epoch_time_str)

        print(display_str, end='\r')

    @staticmethod
    def _get_elapsed_time_str(start_time):
        time_elapsed = dt.now() - start_time
        hours, remainder = divmod(time_elapsed.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return "{:.0f}h {:.0f}m {:.0f}s".format(hours, minutes, seconds)

    @staticmethod
    def _sum_gradients(model):
        grad_sum = 0.0
        for p in model.parameters():
            grad_sum += p.grad.abs().sum()
        return grad_sum

    @staticmethod
    def _buffered_str(str):
        line_length = 100
        buffer_size = line_length - len(str)
        buffer = ' ' * buffer_size
        return str + buffer
