from copy import deepcopy
import numpy as np
from scipy.io import arff
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
from itertools import cycle
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, FastICA, PCA, NMF



__all__ = ['read_data', 'log_scale', 'std_scale', '_ica', '_tsne', '_truncatedSVD', '_nmf', 'plot_res', 'pca', '_pca']

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

def plot_res(X = None, label = None, K = 3, tsne = False, save = False, alg=None):
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
    if not save: plt.show()
    else: plt.savefig('fig/'+alg+str(K)+'.png')

def elbow_method(cost, K):
    k = None

    return k

