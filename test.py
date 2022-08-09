import numpy as np
from pyparsing import col
from sklearn.metrics import pairwise_distances
from scipy.io import arff
import pandas as pd 
from sklearn.metrics import silhouette_score
from model import Kmeanspp, KMedoids, kmeans, agg_clustering, kmedoids
from utils import *
import re

# file_name='data/User Groups Dataset-131atts.arff'
# data, meta=arff.loadarff(file_name)


# # print(meta)
# group = {b'High Consumption' : 0, b'Low Consumption' : 1, b'Medium Consumption' : 2}

# train_x = []
# label = []
# for i in data:
#     i = np.array(list(i))
#     train_x.append(i[2:-2])
#     label.append(i[-1])

# label = np.array([group[i] for i in label])
# for i in range(3):
#     print(len(label[label == i]))
# train_x_ori = np.array(train_x, dtype = np.float32)

# # scale = StandardScaler()
# # train_x = scale.fit_transform(train_x_ori)

# scale = my_scaler()
# data = 1.0 + train_x_ori 
# train_x =  scale.fit(data)


file =  "data/new_matrix.pkl"
data, mapping = read_data(file)


train_x = _ica(data, n_dim = 30)
# train_x = _truncatedSVD(data, n_dim=10)

clu = 3
print("kmeans")
t1 = kmeans(clu, max_iter = 10000, metric = 'cosine')
out1 = t1.fit(train_x)
plot_res(train_x, out1, K = clu)
score1 = silhouette_score(train_x, out1)
print(score1)

# # print("kmedoids")
# # t2 = KMedoids(3, max_iter = 10000, metric = 'cosine', tol = 0.001)
# # out2, cen = t2.fit(train_x)
# # plot_res(train_x, out2, K = 3)
# # score2 = silhouette_score(train_x, out2)
# # print(score2)
# # print(cen)

print("kmedoids")
t2 = kmedoids(clu, metric = 'cosine', n_trials = 100, max_iter = 1000, tol = 0.01)
out2, _, _ = t2.fit(train_x)
plot_res(train_x, out2, K = clu)
score2 = silhouette_score(train_x, out2)
print(score2)

print("kmeans++")
t3 = Kmeanspp(k = clu, max_iter=10000, metric = 'euclidean')
out3 = t3.fit(train_x).labels


plot_res(train_x, out3, K = clu)
score3 = silhouette_score(train_x, out3)
print(score3)


# print("agg")
# t4 = agg_clustering()
# out4 = t4.fit(train_x)
# out4_ = t4.cluster_merge(train_x, out4, threshold = 0.1)
# score4 = silhouette_score(train_x, out4)
# score4_ = silhouette_score(train_x, out4_)
# t4.plot_res(train_x, out4_)
# print(np.unique(out4).shape[0])
# print(np.unique(out4_).shape[0])
# print(score4)
# print(score4_)


# out1 = pd.DataFrame(out1)
# out1.to_csv("data/result.csv")
