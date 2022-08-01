import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Kmeanspp:
    """K-Means++ Clustering Algorithm"""

    def __init__(self, k, centers=None, cost=None, iter=None, labels=None, max_iter=1000):
        """Initialize Parameters"""

        self.max_iter = max_iter
        self.k = k
        self.centers = np.empty(1)
        self.cost = []
        self.iter = 1
        self.labels = np.empty(1)

    def calc_distances(self, data, centers, weights):
        """Distance Matrix"""
        # a : m * n, b : c * n
        # dist : m * c
        # a中m个样本分别与c个样本的距离
        # min_distance : (m ), 与之最近的聚簇
        distance = pairwise_distances(data, centers) ** 2
        min_distance = np.min(distance, axis=1)
        D = min_distance * weights
        return D

    def initial_centers_Kmeansapp(self, data, k, weights):
        """Initialize centers for K-Means++"""

        centers = []
        centers.append(random.choice(data))
        while (len(centers) < k):
            distances = self.calc_distances(data, centers, weights)
            prob = distances / sum(distances)
            c = np.random.choice(range(data.shape[0]), 1, p=prob)
            centers.append(data[c[0]])
        return centers

    def fit(self, data, weights=None):
        """Clustering Process"""

        if weights is None: weights = np.ones(len(data))
        if type(data) == pd.DataFrame: data = data.values
        nrow = data.shape[0]
        self.centers = self.initial_centers_Kmeansapp(data, self.k, weights)

        while (self.iter <= self.max_iter):
            distance = pairwise_distances(data, self.centers) ** 2
            self.cost.append(sum(np.min(distance, axis=1)))
            self.labels = np.argmin(distance, axis=1)
            centers_new = np.array([np.mean(data[self.labels == i], axis=0) for i in np.unique(self.labels)])

            ## sanity check
            if (np.all(self.centers == centers_new)): break
            self.centers = centers_new
            self.iter += 1

        ## convergence check
        if (sum(np.min(pairwise_distances(data, self.centers) ** 2, axis=1)) != self.cost[-1]):
            warnings.warn("Algorithm Did Not Converge In {} Iterations".format(self.max_iter))
        return self

def plot_res(X = None, out = None):
    colors = ['r', 'g', 'b']
    ax = plt.figure().subplots(1, 1)
    ax1 = ax
    for k in range(3):
        data = X[np.where(out == k)[0], :]
        print(data.shape[0])
        x, y = data[:, 0], data[:, 1]
        ax1.scatter(x, y, c=colors[k])
    plt.show()

if __name__ == '__main__':
    trainer = Kmeanspp(3)
    n_samples = 200
    centers = [[2, 2], [1, -2]]
    X, labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1, random_state=0)
    print(X)
    trainer.fit(X)
    print(trainer.labels)
    plot_res(X, trainer.labels)

