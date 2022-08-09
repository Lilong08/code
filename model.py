# -*- coding: utf-8 -*-
from joblib import parallel_backend
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from itertools import cycle
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
import numpy as np
import random
import time
from copy import deepcopy
import warnings


# ----------------  kmeas++ ----------------------

class Kmeanspp:
    """K-Means++ Clustering Algorithm"""

    def __init__(self, k, max_iter=10000, metric = "euclidean"):
        """Initialize Parameters"""

        self.max_iter = max_iter
        self.k = k
        self.cost = []
        self.iter = 1
        self.metric = metric

    def calc_distances(self, data, centers, weights):
        """Distance Matrix"""
        # a : m * n, b : c * n
        # dist : m * c
        # a中m个样本分别与c个样本的距离
        # min_distance : (m ), 与之最近的聚簇
        distance = pairwise_distances(data, centers, metric = self.metric)
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

        while True:
            distance = pairwise_distances(data, self.centers, metric = self.metric)
            self.cost.append(sum(np.min(distance, axis=1)))
            self.labels = np.argmin(distance, axis=1)
            centers_new = np.array([np.mean(data[self.labels == i], axis=0) for i in np.unique(self.labels)])

            

            # sanity check
            if (np.all(self.centers == centers_new)): 
                print("clusters no change")
                break
            self.centers = centers_new
            self.iter += 1
            if(self.iter > self.max_iter):
                print("max iteration")
                break

        # convergence check
        if (sum(np.min(pairwise_distances(data, self.centers), axis=1)) != self.cost[-1]):
            warnings.warn("Algorithm Did Not Converge In {} Iterations".format(self.max_iter))
        return self.labels, self.cost[-1]

# -------------------- kmeans ----------------------

class kmeans:
    ''' kmeans for clustering '''

    '''
    k: number of clusters
    max_iter: max iteration numbers
    '''

    def __init__(self, k, max_iter=1000, metric = "euclidean"):
        self.max_iter = max_iter
        self.k = k
        self.metric = metric
        self.cost = .0

    def fit(self, X):
        labels = self.kmeans_run(X, self.k, self.max_iter, dist = np.mean)
        return labels, self.cost

    def kmeans_run(self, flows, k, max_iter=None, dist=np.mean):
        """
        Calculates k-means clustering with specific metric.

        :param flows: numpy array of shape (n, m), where r is the number of rows
        :param k: number of clusters
        :param dist: for center choose
        :return: numpy array of shape (k, 2)
        """
        rows = flows.shape[0]
        cnt = 0

        distances = np.empty((rows, k))
        # last_clusters = np.zeros((rows,))
        last_clusters = np.ones((rows,))

        np.random.seed()

        # the Forgy method will fail if the whole array contains the same rows
        clusters = flows[np.random.choice(rows, k, replace=False)]
        # print(rows)
        # print(clusters)
        
        while True:
            # for row in range(rows):
            #     distances[row] = distance(flows[row], clusters)

            distances = self.calc_distance(flows, clusters)
            nearest_clusters = np.argmin(distances, axis=1)

            if (last_clusters == nearest_clusters).all():
                print("clusters no change")
                break

            ### 确定每个簇的新质心
            _cost = np.zeros((k))
            for cluster in range(k):
                clusters[cluster] = dist(flows[nearest_clusters == cluster], axis=0)
                _cost[cluster] = np.sum(self.calc_distance(flows[nearest_clusters == cluster], clusters=[clusters[cluster]]))

            last_clusters = nearest_clusters
            cnt += 1
            if (cnt > max_iter):
                print("max iteration")
                break
        self.cost = np.sum(_cost)
        return last_clusters

    def distance(self, flow, clusters):
        """
        Calculates the l2 distance between a flow and k clusters
        :param flow: array, (m, )
        :param clusters: numpy array of shape (k, m) where k is the # of clusters and m is the # fof dim
        :return: numpy array of shape(k, 0) where k is the # of clusters
        """
        
        distance =  np.linalg.norm(flow - clusters, axis = 1)
        return distance

    def calc_distance(self, data, clusters):
        """Distance Matrix"""

        dist_mat = pairwise_distances(data, clusters, metric = self.metric)
        return dist_mat


# ----------------- k-medoids ----------------------

class KMedoids(object):
    '''
    KMedoids Clustering
    Parameters
    --------
        n_clusters: number of clusters
        dist_func : distance function
        max_iter: maximum number of iterations
        tol: tolerance
    Methods
    -------
        fit(X): fit the model
            - X: 2-D numpy array, size = (n_sample, n_features)

    '''

    def __init__(self, n_clusters, max_iter=5000, metric = "euclidean", tol=0.001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric

    def fit(self, X, plotit=False, verbose=True):
        
        centers, labels, costs, tot_cost, dist_mat = self.kmedoids_run(
            X, self.n_clusters, max_iter=self.max_iter, tol=self.tol, verbose=verbose)
        # print(labels)
        if plotit:
            fig, ax = plt.subplots(1, 1)
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            if self.n_clusters > len(colors):
                raise ValueError('we need more colors')

            for i in range(len(centers)):
                X_c = X[labels == i, :]
                ax.scatter(X_c[:, 0], X_c[:, 1], c=colors[i], alpha=0.5, s=30)
                ax.scatter(X[centers[i], 0], X[centers[i], 1], c=colors[i], alpha=1., s=250, marker='*')
            plt.show()
        return labels, centers

    def get_init_centers(self, n_clusters, n_samples):
        '''return random points as initial centers'''
        init_ids = []
        while len(init_ids) < n_clusters:
            _ = np.random.randint(0, n_samples)
            if not _ in init_ids:
                init_ids.append(_)
        return init_ids

    def kmedoids_run(self, X, n_clusters, max_iter=1000, tol=0.001, verbose=True):
        '''run algorithm return centers, members, ...'''
        # Get initial centers
        n_samples, n_features = X.shape
        init_ids = self.get_init_centers(n_clusters, n_samples)
        if verbose:
            print(
            'Initial centers are ', init_ids)
        centers = init_ids
        members, costs, tot_cost, dist_mat = self.get_cost(X, init_ids)
        cc, SWAPED = 0, True
        while True:
            print(cc)
            SWAPED = False
            for i in range(n_samples):
                if not i in centers:
                    for j in range(len(centers)):
                        centers_ = deepcopy(centers)
                        centers_[j] = i
                        members_, costs_, tot_cost_, dist_mat_ = self.get_cost(X, centers_)
                        if tot_cost_ - tot_cost < -tol:
                            members, costs, tot_cost, dist_mat = members_, costs_, tot_cost_, dist_mat_
                            centers = centers_
                            SWAPED = True
                            # if verbose:
                            #     print(
                            #     'Change centers to ', centers)
            if cc > max_iter:
                if verbose:
                    print(
                    'max iteration', max_iter)
                break
            if not SWAPED:
                if verbose:
                    print(
                    'clusters no change')
                break
            cc += 1
        return centers, members, costs, tot_cost, dist_mat
    
    def get_cost(self, X, centers_id):
        '''return total cost and cost of each cluster'''
        '''
        costs, size = (len(centers_id), )
        '''
        dist_mat = np.zeros((len(X), len(centers_id)))
        
        center = X[centers_id]
        dist_mat = pairwise_distances(X, center, metric = self.metric)

        mask = np.argmin(dist_mat, axis=1)
        members = np.zeros(len(X))
        costs = np.zeros(len(centers_id))
        for i in range(len(centers_id)):
            mem_id = np.where(mask == i)
            members[mem_id] = i
            costs[i] = np.sum(dist_mat[mem_id, i])
        return members, costs, np.sum(costs), dist_mat

class kmedoids(object):
    '''
    kmedoids clustering
    Parameters
    --------
        K: number of clusters
        metric : distance function
        n_trials : kmedoids run times
        max_iter: maximum number of iterations
        tol: tolerance
    Methods
    -------
        fit(X): fit the model
            - X: 2-D numpy array, size = (n_sample, n_features)
    Return
    ------
        labels:
        centers:
        cost:
    '''
    def __init__(self, k = 3, metric = 'euclidean', n_trials = 10, max_iter = 100, tol = 0.001):
        self.n_cluster = k
        self.metric = metric
        self.n_trials = n_trials
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        labels, centers, costs = self.kmedoids_(X, self.n_cluster, self.n_trials, self.max_iter, self.tol)
        return (labels, centers, costs)

    def calc_distances(self, data, centers, weights):
        # a : m * n, b : c * n
        # dist : m * c
        # a中m个样本分别与c个样本的距离
        # min_distance : (m ), 与之最近的聚簇
        distance = pairwise_distances(data, centers, metric = self.metric)
        min_distance = np.min(distance, axis=1)
        D = min_distance * weights
        return D

    def _get_init_centers(self, data, k, weights):
        '''
        initial center from kmeans++
        '''
        centers = []
        centers.append(random.choice(data))
        while (len(centers) < k):
            distances = self.calc_distances(data, centers, weights)
            prob = distances / sum(distances)
            c = np.random.choice(range(data.shape[0]), 1, p=prob)
            centers.append(data[c[0]])
        return centers

    def get_init_centers(self, X, n_clsters, n_samples):
        center_id =[]
        while(len(center_id) < n_clsters):
            id = np.random.randint(0, n_samples)
            if(id not in center_id): center_id.append(id)
        return X[center_id]


    def update_center(self, X, labels, n_cluster):
        centers = []
        costs = .0
        # rd_choice = []
        for i in range(n_cluster):
            mem_id = np.where(labels == i)[0]
            # if(mem_id.shape[0] == 0): 
            #     mem_id = np.random.choice(X.shape[0], size = 1)

            members = X[mem_id]
            dist = pairwise_distances(members, metric = self.metric)
            cost = np.sum(dist, axis = 1)
            c_id = np.argmin(cost)
            centers.append(members[c_id])
            costs += cost[c_id]

        return centers, costs

    def kmedoids_run(self, X, n_cluster, max_iter, tol):
        n_samples = X.shape[0]

        # center_id = self.get_init_centers(n_cluster, n_samples)
        W = np.ones(n_samples)
        # print("initial centers : ", center_id)
        # centers = X[center_id]
        centers = self._get_init_centers(X, n_cluster, W)

        iter = 0
        last_cost = np.inf
        labels = np.empty(shape = n_samples, dtype = int)

        while True:
            dist_mat = pairwise_distances(X, centers, metric = self.metric)
            labels = np.argmin(dist_mat, axis = 1)
            # print(np.unique(labels).shape[0])
            cur_center, cur_cost = self.update_center(X, labels, n_cluster)
            # if(np.abs(last_cost - cur_cost) < tol):
            #     # diff is smaller the par tol, then accept new center
            #     centers = cur_center
            #     break

            if(cur_cost < last_cost): 
                centers = cur_center
                last_cost = cur_cost
            iter += 1
            if(iter >= max_iter): 
                break
        # print(iter)
        return centers, labels, last_cost

    def kmedoids_(self, X, n_cluster, max_trials, max_iter, tol):

        trial = 0
        centers, labels, cost = self.kmedoids_run(X, n_cluster, max_iter, tol)
        while trial < max_trials:
            centers_, labels_, cost_ = self.kmedoids_run(X, n_cluster, max_iter, tol)
            if(cost_ < cost):
                centers = centers_
                labels = labels_
                cost = cost_
            trial += 1

        return labels, centers, cost


# ----------------- Hierarchical Clustering ----------------------
class agg_clustering:
    '''
    Agglomerative Hierarchical Clustering

    affinity = euclidean (cosine)

    linkage = ward

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
    
    def cluster_merge(self, X, labels, threshold = 0.001):
        # cluster merge from nsdi22
        
        res = labels
        tmp_labels = deepcopy(labels)
        cnt = 0
        while (True):
            labels_ = np.unique(tmp_labels)
            centeroid = []
            mapping = {}
            for idx, i in enumerate(labels_):
                # calculate the centroid of each cluster
                centeroid.append(np.mean(X[np.where(tmp_labels == i)], axis = 0))
                mapping[idx] = i
                
            center = np.array(centeroid)
            
            # distance matrix size = (n, n)
            # n = center.shape[0]
            distance = pairwise_distances(center, metric = 'cosine')
            for i in range(distance.shape[0]):
                distance[i, i] = np.inf
            
            # size = (n, )
            value_min = np.min(distance, axis = 1)
            if(np.min(value_min) > threshold): break
            
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
        # nn = np.unique(labels).shape[0]
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(set(labels), colors):
            # get members
            my_members = labels == k
            # get indexes
            x, y = X[my_members, 0], X[my_members, 1]
            print(x.shape[0])
            plt.plot(x, y, col+'.')
        plt.title('euclidean-ward')
        plt.show()
