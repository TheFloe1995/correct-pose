"""
Attempt to cluster the results of some embedding on a dataset (e.g. from TSNE).
"""

import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.pyplot as plt

# Config
data_file = 'results/00_TSNE/HANDS17_DPREN_ShapeSplit_val_normalized_DEFAULT_combined_50.npy'
########################################################################

data = np.load(data_file)

epsilons = [2.0, 4.0]
clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=1000)
clust.fit(data)

for epsilon in epsilons:
    print('Starting clustering for epsilon {}'.format(epsilon))
    labels = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=epsilon)
    cluster_labels = set(labels)
    for i in cluster_labels:
        mask = labels == i
        cluster = data[mask]
        if i == -1:
            plt.scatter(cluster[:, 0], cluster[:, 1], s=1, alpha=1.0, label=i, color='black')
        else:
            plt.scatter(cluster[:, 0], cluster[:, 1], s=1, alpha=0.2, label=i)
    plt.show()
