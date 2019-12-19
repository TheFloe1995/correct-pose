# Hand Pose Correction with PyTorch

This repository contains code for my Master's thesis: 
**Correcting 3D Hand Poses by Learning a Structural Prior with Graph Neural Networks**

It's a PyTorch implementation, tested with version PyTorch 1.10 and Python 3.7.
Required packages:
* PyTorch
* Scipy
* Matplotlib

## How to get started
Training scripts are located in [run](run). Make sure to have the datasets available in the
required format in [data](data). More details and scripts for generating data are located in 
[data_utils](data_utils). Try to run the [minimal example script](run/minimal_example.py)
first and check it out in detail.

To make sure everything is working on your machine I recommend to run the unit tests in 
[unit_tests](unit_tests) using *pytest*.

After getting familiar with the data format and the training scripts, have a look at the files in 
[training](training).

After executing an experiment, use the [evaluation](notebooks/Evaluation.ipynb) Jupyter Notebook
to get some insights, plots and visualizations.
