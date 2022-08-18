from copy import deepcopy
import numpy as np
from scipy.io import arff
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import scipy
import seaborn as sns
from itertools import cycle
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, FastICA, PCA, NMF
from sklearn.metrics import pairwise_distances



__all__ = ['read_data', 'log_scale', 'std_scale', '_ica', '_tsne', '_truncatedSVD', '_nmf', 'plot_res', 'pca', '_pca', 'knee', '_hc_cut', 'cluster_merge']

def read_data(file_path):
    with open(file_path, "rb") as f:

        data = pickle.load(file=f)
        mapping = data["mapping1"]
        _data = data["matrix"]
        return _data, mapping


def log_scale(X):
    '''
    log scaling from paper(Cloudcluster, Nsdi22)
    X: non-negtive matrix, size = (n_samples, n_features)
    '''
    assert X.ndim == 2
    _X = X + 1.0
    _X = np.log(_X)
    return _X

def std_scale(X):
    '''
    Standard scale from sklearn
    X: input matrix, size = (n_samples, n_features)
    '''
    scale = StandardScaler()
    _X = scale.fit_transform(X)
    return _X

def _ica(X, n_dim, max_iter = 1000, tol = 0.001, whiten = "unit-variance"):
    '''
    FastICA dimensionality reduction
    X: input data, size = (n_samples, n_features)
    return: _X, size = (n_samples, n_dim)
    '''
    transformer=FastICA(n_components=n_dim, whiten=whiten, max_iter=max_iter, tol=tol)
    _X = transformer.fit_transform(X)
    return _X

def _tsne(X, n_dim, n_iter = 1000, lr = 'auto'):
    '''
    TSNE dimensionality reduction
    X: input raw data, size = (n_samples, n_features)
    n_dim: keeped dim, dim <= 3
    return: _X, size = (n_samples, n_dim)
    '''
    _X = TSNE(n_components = n_dim,n_iter = n_iter , learning_rate = lr).fit_transform(X)
    return _X

def _nmf(X, n_dim, random_state = 0):
    '''
    NMF dimensionality reduction
    X: input raw data, size = (n_samples, n_features)
    n_dim: keeped dim
    return: W, size = (n_samples, n_dim)
    '''
    nmf = NMF(n_components=n_dim, random_state=random_state)
    W = nmf.fit_transform(X)
    H = nmf.components_
    return W

def _truncatedSVD(X, n_dim, n_iter = 1000, random_state=43):
    '''
    TruncatedSVD dimensionality reduction
    X: input raw data, size = (n_samples, n_features)
    '''
    _X = TruncatedSVD(n_components=n_dim, n_iter=n_iter, random_state=random_state).fit_transform(X)
    return _X

def _pca(X, n_dim):
    '''
    pcm dimensionality reduction
    X: input raw data, size = (n_samples, n_features)
    '''
    # _X = PCA(n_components=n_dim).fit_transform(X)
    pca = PCA(n_components=n_dim)
    pca = pca.fit(X)
    _X = pca.transform(X)
    evr = pca.explained_variance_ratio_
    return _X, evr

def pca(data, n_dim):
    '''
    pca is O(D^3)
    data: (n_samples, n_features(D))
    :param n_dim: target dimensions
    :return: (n_samples, n_dim)
    '''
    data = data - np.mean(data, axis = 0, keepdims = True)

    cov = np.dot(data.T, data)

    eig_values, eig_vector = np.linalg.eig(cov)
    # print(eig_values)
    indexs_ = np.argsort(-eig_values)[:n_dim]
    picked_eig_values = eig_values[indexs_]
    picked_eig_vector = eig_vector[:, indexs_]
    data_ndim = np.dot(data, picked_eig_vector)
    return data_ndim

def plot_res(X = None, label = None, K = 3, tsne = False, save = False, alg=None, show=False):
    '''
    plot the clustering results
    X: input data, size = (n_samples, n_features)
    label: clustering label, size = (n_samles, ) 
    K: # of cluster
    tsne: use tsne for better visulization
    '''
    assert K == np.unique(label).shape[0]
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'b', 'g', 'r']
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    shape = cycle('+*o.')

    _X = deepcopy(X)
    if tsne:
        _X = _tsne(_X, 2)
    ax = plt.figure().subplots(1, 1)
    for k, color, sp in zip(set(label), colors, shape):
        data = _X[np.where(label == k)[0], :]
        
        print(data.shape[0])
        
        x, y = data[:, 0], data[:, 1]
        ax.scatter(x, y, c = color, marker = sp)
    if show: plt.show()
    if save: plt.savefig('fig/'+alg+str(K)+'.png')



def _hc_cut(n_clusters, children, n_leaves):
    from heapq import heapify, heappop, heappush, heappushpop
    import _hierarchical_fast as _hierarchical  # type: ignore
    """Function cutting the ward tree for a given number of clusters.

    Parameters
    ----------
    n_clusters : int or ndarray
        The number of clusters to form.

    children : ndarray of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    n_leaves : int
        Number of leaves of the tree.

    Returns
    -------
    labels : array [n_samples]
        Cluster labels for each point.
    """
    if n_clusters > n_leaves:
        raise ValueError(
            "Cannot extract more clusters than samples: "
            "%s clusters where given for a tree with %s leaves."
            % (n_clusters, n_leaves)
        )
    # In this function, we store nodes as a heap to avoid recomputing
    # the max of the nodes: the first element is always the smallest
    # We use negated indices as heaps work on smallest elements, and we
    # are interested in largest elements
    # children[-1] is the root of the tree

    # node is a minimum heap
    # heap top is the largest node(-value)
    nodes = [-(max(children[-1]) + 1)]
    for _ in range(n_clusters - 1):
        # As we have a heap, nodes[0] is the smallest element
        these_children = children[-nodes[0] - n_leaves]
        # Insert the 2 children and remove the largest node
        heappush(nodes, -these_children[0])
        heappushpop(nodes, -these_children[1])
    label = np.zeros(n_leaves, dtype=np.intp)
    for i, node in enumerate(nodes):
        label[_hierarchical._hc_get_descendent(-node, children, n_leaves)] = i
    return label

def knee(data):
    '''
    plot CDF of inconsistency
    inc: np.ndarry or List
    '''
    norm_cdf = scipy.stats.norm.cdf(data)
    sns.lineplot(data, norm_cdf)
    plt.hist(data, bins=10000 ,cumulative=True, histtype='step', density=True)
    plt.show()

def cluster_merge(X, labels, threshold = 0.001):

    '''cluster merge from nsdi22'''
    
    res = labels
    tmp_labels = deepcopy(labels)
    cnt = 0
    while (True):
        labels_ = np.unique(tmp_labels)
        centeroid = []
        mapping = {}
        for idx, i in enumerate(labels_):
            # calculate the centroid of each cluster
            centeroid.append(np.mean(X[np.where(tmp_labels == i)[0]], axis = 0))
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
            
        # merge cluste x to y
        tmp_labels[np.where(tmp_labels == x_label)[0]] = y_label
        res = tmp_labels
        cnt += 1

    print("%d cluster(s) merged"%cnt)    
    return res

