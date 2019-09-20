import os
import torch
import copy

from data_utils import datasets
from training.solver import Solver
from data_utils import distorters


# In a single experiment an arbitrary number of models can be trained sequentially. First the
# experimental setup is configured (setting base parameters and hyperparameter options) and then
# the whole process can be started using the run() method.
# Only a single dataset per experiment is supported.
# The results are stored to the 'results' directory. The class creates a new subfolder (experiment
# directory) equal to the experiment name (specified in the config). Under this directory it creates
# a results directory for each distinct hyperparamter combination (using increasing integers as
# names) and adds another layer of repetition subdirectories for each repetition.
# The information that is saved:
#   - config.pt: experiment config and solver config (placed only once in experiment directory)
#   - params.pt: hyperparameters in each results directory
#   - log.pt: training log for a single training session (loss, intermediate evaluations)
#   - eval.pt: final results (all metrics, per sample)
#   - weights.pt: final weights of the model (from the best epoch)
#   - examples.pt: example predictions for some selected samples from the validation set collected
#                  during training for later visualization
class Experiment:
    def __init__(self, experiment_config, solver_config, base_hyperparams):
        self.config = {
            'experiment': experiment_config,
            'solver': solver_config,
        }
        self.base_hyperparams = base_hyperparams
        self.options = {}

    # To perform a grid search register the desired parameter by it's name and specify a list of
    # values that should be tried in different training sessions.
    # IMPORTANT: The param_name parameter must be tuple of length 2. The hyperparameters dictionary
    # can be nested up to 2 layers. If the hyperparameter is on the highest level, the first entry
    # of the tuple needs to be None and the second element contains the dictionary key. If the
    # hyperparameter is contained in a sub dictionary B contained in the hyperparameter dictionary
    # A, the first parameter specifies the key in A for B and the second parameter contains the key
    # from B for the target parameter.
    def add_options(self, param_name, values):
        self.options[param_name] = values
        self.config['solver']['interest_keys'].append(param_name)

    def run(self):
        distorter = self.base_hyperparams['distorter'](self.base_hyperparams['distorter_args'])
        if self.config['experiment']['normalizer'] is not None:
            train_data = datasets.NormalizedPairedPoseDataset(
                self.config['experiment']['train_set'],
                distorter,
                self.config['experiment']['normalizer'],
                self.config['experiment']['use_preset'],
                self.config['experiment']['train_set_size'],
                self.config['experiment']['target_device'])
            val_data = datasets.NormalizedPairedPoseDataset(
                self.config['experiment']['val_set'],
                distorters.NoDistorter(),
                self.config['experiment']['normalizer'],
                True,
                self.config['experiment']['val_set_size'],
                self.config['experiment']['target_device'])
        else:
            train_data = datasets.PairedPoseDataset(self.config['experiment']['train_set'],
                                                    distorter,
                                                    self.config['experiment']['use_preset'],
                                                    self.config['experiment']['train_set_size'],
                                                    self.config['experiment']['target_device'])
            val_data = datasets.PairedPoseDataset(self.config['experiment']['val_set'],
                                                  distorters.NoDistorter(),
                                                  True,
                                                  self.config['experiment']['val_set_size'],
                                                  self.config['experiment']['target_device'])

        self.config['experiment']['train_set_size'] = len(train_data)
        self.config['experiment']['val_set_size'] = len(val_data)

        experiment_dir = os.path.join('results', self.config['experiment']['name'])
        os.mkdir(experiment_dir)
        torch.save(self.config, os.path.join(experiment_dir, 'config.pt'))

        solver = Solver(self.config['solver'], train_data, val_data)
        combinations_of_configs = self._generate_combinations()

        for i, hyperparams in enumerate(combinations_of_configs):
            print('\n\n' + '#' * 100)
            print('START OF SESSION {}/{}'.format(i + 1, len(combinations_of_configs)))

            results_dir = os.path.join(experiment_dir, str(i))
            os.mkdir(results_dir)
            torch.save(hyperparams, os.path.join(results_dir, 'params.pt'))

            distorter = hyperparams['distorter'](hyperparams['distorter_args'])
            train_data.distorter = distorter

            for j in range(self.config['experiment']['n_repetitions']):
                print('\nRepetition {}/{}  ({}):'.format(j + 1,
                                                         self.config['experiment']['n_repetitions'],
                                                         self.config['experiment']['name']))
                print('*' * 50)

                model = self._create_model_and_normalizer(hyperparams)

                log, eval_results, weights, example_predictions = solver.train(model, hyperparams)

                repetition_dir = os.path.join(results_dir, str(j))
                os.mkdir(repetition_dir)

                torch.save(log, os.path.join(repetition_dir, 'log.pt'))
                torch.save(eval_results, os.path.join(repetition_dir, 'eval.pt'))
                torch.save(weights, os.path.join(repetition_dir, 'weights.pt'))
                torch.save(example_predictions, os.path.join(repetition_dir, 'examples.pt'))

        print('\nExperiment >> {} << finished.\n'.format(self.config['experiment']['name']))

    # From the registered options this functions generates all possible combinations. Be careful,
    # the number of training runs increases exponentially!
    def _generate_combinations(self):
        hyper_param_configs = [self.base_hyperparams]
        for (sub_dict, param_name), values in self.options.items():
            new_list = []
            for i, config in enumerate(hyper_param_configs):
                for val in values:
                    new_config = copy.deepcopy(config)
                    if sub_dict is None:
                        new_config[param_name] = val
                    else:
                        new_config[sub_dict][param_name] = val
                    new_list.append(new_config)
            hyper_param_configs = new_list
        return hyper_param_configs

    def _create_model_and_normalizer(self, hyperparams):
        model = hyperparams['model'](hyperparams['model_args'])
        model.to(self.config['experiment']['target_device'])

        if self.config['experiment']['init_weights_path'] is not None:
            weights = torch.load(self.config['experiment']['init_weights_path'],
                                 map_location=self.config['experiment']['target_device'])
            model.load_state_dict(weights)

        return model
