# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:03:46 2022

@author: A109021
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

#def show_dataset(X, Y):
#    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
#    ax.grid()
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    for i in range(nb_samples):
#        if Y[i] == 0:
#            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
#        else:
#            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')
#    plt.show()

#X, Y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

#show_dataset(X, Y)


n_jobs=multiprocessing.cpu_count() 