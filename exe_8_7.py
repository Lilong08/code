from hashlib import new
from operator import truediv
from tkinter import N
from turtle import distance
from types import new_class
from utils import *
import re
from model import kmeans, kmedoids, Kmeanspp, agg_clustering, _linkage
from scipy.io import arff
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import make_blobs

import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


file =  "data/new_matrix.pkl"
data, mapping = read_data(file)
pattern = re.compile(r'192\.168\.(.+)')

new_data = []
ip = {}
cnt = 0
for idx, raw in enumerate(data):

    if re.match(pattern, mapping[idx]) is not None:
        new_data.append(raw)
        ip[cnt] = mapping[idx]
        cnt += 1

new_data = np.array(new_data)

X = log_scale(new_data)
# X = new_data

# X = _truncatedSVD(X, 10)
# X = _ica(X, 10)
# X = _nmf(X, 10)
X, _ = _pca(X, 0.90)




# make samples
# n_samples = 2000
# centers = [[2, 2], [-1, -1], [1, -2]]
# X, labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1, random_state=0)


# test for kmeans\kmedoids\kmeans++
res = []
for clu in range(3, 11):
    kk = clu
    print("kk = ", kk)
    t1 = kmeans(k = kk, max_iter = 10000, metric='cosine')
    t2 = kmedoids(kk, 'cosine', 10, 100, tol=0.001)
    t3 = Kmeanspp(kk, 10000, 'cosine')
    t4 = agg_clustering(dist_threshold=0.1)
    out1, c1 = t1.fit(X)
    out2, _, c2= t2.fit(X)
    out3, c3= t3.fit(X)

    print(c1)
    print(c2)
    print(c3)


    print('------show result--------')
    plot_res(X, out1, K=kk, tsne=True, save=False, alg='kmeans', show=False)
    plot_res(X, out2, K=kk, tsne=True, save=False, alg='kmedoids', show=False)
    plot_res(X, out3, K=kk, tsne=True, save=False, alg='kmeans++', show=False)

    sc1 = silhouette_score(X, out1, metric='cosine')
    sc2 = silhouette_score(X, out2, metric='cosine')
    sc3 = silhouette_score(X, out3, metric='cosine')
    # # print(sc1)
    # # print(sc2)

    print(sc1, sc2, sc3)
    res.append([sc1, sc2, sc3])
print(res)
df = pd.DataFrame(data=res)
df.to_csv('data/res.csv')


# # test for Hierarchical Clustering

# trainer = agg_clustering(dist_threshold=82)
# out, _ = trainer.fit(X)
# knee(_)
# print('----------------Hierarchical Clustering ---------------')
# plot_res(X, out, K = np.unique(out).shape[0], tsne=True, show=True)
# out5 = cluster_merge(X, out, 0.15)
# plot_res(X, out5, K = np.unique(out5).shape[0], tsne=True, show=True)
# sc4 = silhouette_score(X, out, metric='euclidean')
# sc5 = silhouette_score(X, out5, metric='euclidean')
# print(sc4, sc5)




# # dfs to check if a child node has a higher inconsistency value than its parent node does
# def dfs(children, node, par, inc, p_i):
#     if node < n_samples: return True
#     idx = node - n_samples
#     p = inc[idx]
#     if p > par: 
#         print(p_i, node)
#         return False

#     lc = children[idx][0]
#     rc = children[idx][1]
#     l = dfs(children, lc, p, inc, node)
#     r = dfs(children, rc, p, inc, node)
#     if l and r: return True
#     return False
# for i in _idx:
#     if dfs(c, c[i][0], _[i], _, i + n_samples) and dfs(c, c[i][1], _[i], _, i + n_samples):continue
#     else: print("error", i)



# # test for linkage with inconsistency
# dist_threshold = 7.2
# # out4 = _hc_cut(n_clusters, children, n_leaves)
# out4, _= _linkage(dist_threshold).fit(X)
# knee(_)

# print('----------linkage----------')
# plot_res(X, out4, K = np.unique(out4).shape[0], tsne=True, show=True)
# out5 = cluster_merge(X, out4, 0.25)
# plot_res(X, out5, K = np.unique(out5).shape[0], tsne=True, show=True)
# sc4 = silhouette_score(X, out4, metric='euclidean')
# sc5 = silhouette_score(X, out5, metric='euclidean')
# print(sc4, sc5)
# print(len(np.unique(out4)), len(np.unique(out5)))


# find the threshold of merging

# import matplotlib.pyplot as plt
# mt = np.linspace(0.01, 1, 20)
# res = []
# for i in mt:
#     out = t4.cluster_merge(X, out4, i)
#     if np.unique(out).shape[0] == 1: 
#         res.append(1.)
#         continue
#     sc5 = silhouette_score(X, out, metric='euclidean')
#     res.append(sc5)

# ax = plt.figure().subplots(1, 1)
# ax.scatter(mt, res)
# plt.show()




# '''calculate precision from labeled dataset'''
# file_name='data/User Groups Dataset-131atts.arff'
# raw_data, meta=arff.loadarff(file_name)

# # print(meta)
# group = {b'High Consumption' : 0, b'Low Consumption' : 1, b'Medium Consumption' : 2}

# lb = {}
# ct = np.zeros(3)
# for i in raw_data:
#     i = np.array(list(i))
#     ii = i[1].decode("utf-8")
#     lb[ii] = group[i[-1]]
#     ct[group[i[-1]]] += 1


# print(ct)


# def res_analysis(label, y, ip, k):
#     cluster = []
#     _label = np.empty((k))

#     for i in range(k):
#         cluster.append(np.where(label == i)[0])
#         cnt = np.zeros((k))
#         for j in cluster[-1]:
#             cnt[y[ip[j]]] += 1

#         print(cnt)
#         _label[i] = np.argmax(cnt)
    

# print('----kmeans----')
# res_analysis(out1, lb, ip, 3)
# print('----kmedoids----')
# res_analysis(out2, lb, ip, 3)
# print('----kmeans++----')
# res_analysis(out3, lb, ip, 3)