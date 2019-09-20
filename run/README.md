# Run
Directory for scripts to run trainings, to test models or whatever. Only place runnable scripts,
no packages.

Run with working directory set to repository root.

In order to minimize the amount of parameters that have to be changed manually every time a new 
experiment is started, it's recommended to maintain separate run files for general training 
settings, e.g. training a specific architecture on a specific dataset.

## Experiment Configuration
An experiment is sequence of training sessions. In each training session, a single model is trained
with a specific set of hyperparameters. There are a lot of different parameters to configure the
training process. At the beginning of each training-script comes a set of dictionaries that define
the basis of parameters that should be used. They are divided into 3 sub-categories:
* **Experiment configuration**: Parameters related to the whole experiment, i.e. they are required
    before any solver or training session is instantiated.
* **Solver configuration**: Parameters related to how a model is trained, basically required when
    instantiating a solver object.
* **Hyperparameters**: Everything else that is directly related to the model or the optimization 
    process and thus potentially influences the final performance.

For the hyperparameters, an easy grid search can be configured in the 2nd section of a training
script. First specify the different values that should be used for training in form of a list. Then
register these *options* in the experiment instance. You can also register options for multiple
hyperparameters at once. The experiment automatically generates all possible combinations and then 
starts a training session for combination sequentially and saves the results to disk. To just train
a single model, specify the desired hyperparameters in the base configuration and don't register any
options. See one of the existing training scripts for an example and some further details.

### All parameters explained
#### Experiment Configuration:
* **name**: Meaningful name that also determines the name of the folder created on disk
* **train_set**: Name of the training set located in the *data* directory, e.g. 
    'HANDS17_DPREN_SubjClust_train' (no file ending required, leave out suffixes like '_poses') 
* **val_set**: Same as above
* **train_set_size**: `None` for taking the whole set or an `int` to take just the first n samples
* **val_set_size**: Same as above
* **use_preset**: `bool`, whether to load predictions from disk in addition to the labels or not 
    (set it to `False` if a powerful distorter is used to generate input samples)
* **normalizer**: Normalizer class that should be used to normalize the samples
* **target_device**: `torch.device` object (not just `int` or `str`)
* **n_repetitions**: `int` >= 1: how often each training session should be repeated (without 
    changing any parameters) 
* **init_weights_path**: Path to a file with weights if training is continued.

### Solver Configuration:
* **log_frequency**: `int`, how often to write intermediate validation results to log per epoch
* **log_loss**: `bool`, whether to write the loss to the log as well (for every batch)
* **log_grad**: `bool`, whether to write the average norm of the gradient vector to the log
* **verbose**: `bool`, for more information (but also more spam) while training (output might look 
    a bit weird)   
* **show_plots**: `bool`, plot the predicted pose + label occasionally during training
* **num_epochs**: `int`
* **batch_size**: `int`
* **interest_keys**: [], used internally only, just initialize it as an empty list, after executing 
    an experiment this list contains the dictionary keys of the parameters that have been changed
    during the experiment (the parameters of interest for the grid search) 
* **val_example_indices**: `list` of `int`, specifying example in the training set for which
    occasionally predictions are saved during training in order to visualize later how the
    corrections get gradually better.
* **val_example_subset**: `str`, name of the subset from which to take the examples above or 
    'DEFAULT' if the dataset has no subsets

### Hyperparameters
* **model**: class (type name) of the model
* **model_args**: `dict`, depending on the model chosen above (see the class definitions for detail)
* **loss_function**: class (type name) of the loss function (see networks.losses for detail)
* **loss_space**: `str`, 'default' or 'original', whether to apply the loss in original space 
    (after denormalization) or in the normalized space (default). If no normalization is used at all
    this must be set to 'default'. I know, that's ugly.
* **eval_space**: same as above for the intermediate evaluations (logs) 
* **optimizer**: class (type name) of the optimizer
* **optimizer_args**: `dict`, names of the constructor parameters of the above stated optimizer as
    keys
* **scheduler**: class (type name) of the learning rate scheduler
* **scheduler_requires_metric**: `bool`, some schedulers (e.g. PyTorch's 
    ReduceLearningRateOnPlateau) 
    need to monitor a metric, indicate this by setting it to `True` for the internal solver logic
* **scheduler_args**: `dict`, names of the constructor parameters of the above stated scheduler as
    keys
* **distorter**: class (type name) of the distorter used during training from 
    `data_utils.distorters` (for no distortion use `data_utils.distorters.NoDistorter`)
* **distorter_args**: `dict`, depending on the distorter chosen above (see class definitions for 
    details)
* **augmenters**: `list` of augmenter objects (already instantiated) from `data_utils.augmenters` 
    that are used for data augmentation (in that order)
