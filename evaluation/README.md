# Evaluation
This package contains code related to everything related to evaluating and analyzing the performance
and the errors of the models.

## Difference between errors and metrics
Errors can be positive or negative while a large absolute value means having a large error. Errors 
can have arbitrary shapes (e.g. having an error for each joint or for each joint coordinate). 

Metrics are always scalar for a pair of samples and the optimum is either achieved by maximizing or
minimizing that metric. 