# Hand Pose Correction with PyTorch

This repository contains code for my Master's thesis.
**TODO**: Add title and possibly abstract or link.

It's a PyTorch implementation, tested with version PyTorch 1.10 and Python 3.7.
**TODO**: Adda requirements.txt or manually list required packages.

## How to get started
Training scripts are located in [run](run). Make sure to have the datasets available in the
required format in [data](data). Mor details and scripts for generating data are located in 
[data_utils](data_utils). 

To make sure everything is working on your machine I recommend to run the unit tests in 
[unit_tests](unit_tests) using *pytest*.

After getting familiar with the data format and the training scripts, have a look at the files in 
[training](training).

After executing an experiment, use the [evaluation](notebooks/Evaluation.ipynb) Jupyter Notebook
to get some insights, plots and visualizations.