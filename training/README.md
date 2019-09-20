# Training
Package containing classes and functions related to the training process.

There are 3 main classes:
* **TrainingSession**: Training instance of a sinlge model with a fixed set of hyperparameters, that
    encapsulates low level logic related to calling the model's forward and backward pass, computing
    the gradients, performing parameter updates and so on.
* **Solver**: Class for higher level training logic. A solver is associated with a single dataset 
    but it can be used to train multiple models with different hyperparameters in a row. Furthermore
    it handles all the logging and printing.
* **Experiment**: Manages the training of a whole set of different models with variable 
    hyperparameters (grid search) and saves the settings, results, weights and logs to disk.