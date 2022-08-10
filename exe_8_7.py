from hashlib import new
from utils import *
import re
from model import kmeans, kmedoids, Kmeanspp, agg_clustering
from scipy.io import arff
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from pyclust import KMedoids
import pandas as pd

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

# X = _truncatedSVD(new_data, 10)
# X = _ica(X, 10)
X = _nmf(X, 10)




# X, evr= _pca(X, 0.9)
# print(evr)
# print(np.sum(evr))

# n_samples = 2000
# centers = [[2, 2], [-1, -1], [1, -2]]
# X, labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1, random_state=0)

res = []
for clu in range(3, 11):
    kk = clu
    print("kk = ", kk)
    t1 = kmeans(k = kk, max_iter = 1000, metric='cosine')
    t2 = kmedoids(kk, 'cosine', 10, 100, tol=0.001)
    t3 = Kmeanspp(kk, 10000, 'cosine')
    t4 = agg_clustering(dist_threshold=0.1)
    out1, c1 = t1.fit(X)
    out2, _, c2= t2.fit(X)
    out3, c3= t3.fit(X)

    print(c1)
    print(c2)
    print(c3)

    # out4 = t4.fit(X)
    # print('----------agg')
    # plot_res(X, out4, K = np.unique(out4).shape[0], tsne=True)
    # out5 = t4.cluster_merge(X, out4, 0.1)
    # plot_res(X, out5, K = np.unique(out5).shape[0], tsne=True)
    # X = new_data
    # print('------other')
    plot_res(X, out1, K=kk, tsne=True, save=False, alg='kmeans')
    plot_res(X, out2, K=kk, tsne=True, save=False, alg='kmedoids')
    plot_res(X, out3, K=kk, tsne=True, save=False, alg='kmeans++')

    sc1 = silhouette_score(X, out1)
    sc2 = silhouette_score(X, out2)
    sc3 = silhouette_score(X, out3)
    # # print(sc1)
    # # print(sc2)

    print(sc1, sc2, sc3)
    # res.append([sc1, sc2, sc3])

# df = pd.DataFrame(data=res)
# df.to_csv('data/res.csv')

'''below code is useless'''
# file_name='data/User Groups Dataset-131atts.arff'
# raw_data, meta=arff.loadarff(file_name)

# # print(meta)
# group = {b'High Consumption' : 0, b'Low Consumption' : 1, b'Medium Consumption' : 2}

# lb = {}
# for i in raw_data:
#     i = np.array(list(i))
#     ii = i[1].decode("utf-8")
#     lb[ii] = group[i[-1]]

# def res_analysis(label, y, ip, k):
#     cluster = []
#     _label = np.empty((k))
#     true_y = np.array([y[k] for k in y.keys()])
#     for i in range(k):
#         cluster.append(np.where(label == i)[0])
#         cnt = np.zeros((k))
#         for j in cluster[-1]:
#             cnt[y[ip[j]]] += 1
#         _label[i] = np.argmax(cnt)
#     print(_label)

#     if(len(set(_label)) == 3):
#         for idx, clu in enumerate(cluster):
#             label[clu] == _label[idx]
#         cm = confusion_matrix(true_y, label)
#         plt.matshow(cm, cmap=plt.cm.Reds)
#         plt.show()
# res_analysis(out3, lb, ip, 3)





