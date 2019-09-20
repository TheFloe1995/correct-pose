# Data
Place the dataset files into this directory. For a description of the file format, see 
[data utils](../data_utils/README.md).

It is recommended to place raw dataset files into subdirectories in this folder or use symlinks.

## Subdirectories
The following subdirectories are required by some parts of the code:
* **clusterings**: Results of clusterings applied to some datasets for further use.
* **distortions**: Errors (prediction - label) of some datasets used e.g. by a 
[PredefinedDistorter](../data_utils/distorters.py).
* **knn**: Results of a KNN-analysis of some datasets used e.g. by the 
[KNNPredefinedDistorter](../data_utils/distorters.py).
* **subset_indices**: Files containing indices of samples to split datasets into train/val/test 
    subsets.
* **unit_test**: Files required for some unit tests to work.
