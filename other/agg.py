# -*- coding: utf-8 -*-
from copy import deepcopy
from dis import dis
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from itertools import cycle
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
import numpy as np

class agg_clustering:

    '''
    Agglomerative Hierarchical Clustering

    affinity = euclidean
    (euclidean:欧式距离
    minkowski:明氏距离
    chebyshev:切比雪夫距离
    canberra:堪培拉距离)

    linkage = ward
    (single:  MIN
    ward：沃德方差最小化
    average：UPGMA
    complete：MAX)

    '''
    def __init__(self, aff = 'euclidean', link = 'ward', dist_threshold = 3.0, n_cluster = None):
        self.aff = aff
        self.link = link
        self.dist_threshold = dist_threshold
        self.n_cluster = n_cluster
        self.trainer = AgglomerativeClustering(n_clusters = self.n_cluster, affinity=self.aff, linkage=self.link, distance_threshold=self.dist_threshold)
    

    def fit(self, X):
        labels = self.trainer.fit_predict(X)
        return labels
    
    def cluster_merge(self, X, labels, threshold = 0.0001):
        # calculate the centroid of each cluster
        
        res = labels
        tmp_labels = deepcopy(labels)
        cnt = 0
        while (True):
            labels_ = np.unique(tmp_labels)
            centeroid = []
            mapping = {}
            for idx, i in enumerate(labels_):
                centeroid.append(np.mean(X[np.where(tmp_labels == i)], axis = 0))
                mapping[idx] = i
                
            center = np.array([centeroid])
            
            # distance matrix size = (n, n)
            # n = center.shape[0]
            distance = pairwise_distances(center, metric = 'cosine')
            for i in range(distance.shape[0]):
                distance[i, i] = np.inf
            
            # size = (n, )
            value_min = np.min(distance, axis = 1)
            if(value_min > threshold): break
            
            idx_min = np.argmin(distance, axis = 1)
            x = np.argmin(value_min)
            y = idx_min[x]
            x_label = mapping[x]
            y_label = mapping[y]
            
            # merge clustes x to y
            tmp_labels[np.where(tmp_labels == x_label)] = y_label
            res = tmp_labels
            cnt += 1

        print("%d cluster(s) merged"%cnt)
        return res

    def plot_res(self, X, labels):
        # plot the result

        plt.figure(1)
        plt.clf()
        nn = np.unique(labels).shape[0]
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(0, nn), colors):
            ##根据lables中的值是否等于k，重新组成一个True、False的数组
            my_members = labels == k
            ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.title('euclidean-ward')
        plt.show()