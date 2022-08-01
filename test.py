import secrets
import numpy as np
from pyparsing import col
from sklearn.metrics import pairwise_distances
import pandas as pd
from scipy.io import arff
import pandas as pd 
import matplotlib.pyplot as plt
from scaler import my_scaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pickle
from itertools import cycle
from scaler import my_scaler
from model import Kmeanspp, KMedoids, kmeans, agg_clustering
from dim_reduction import pca


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
# train_x_ori = np.array(train_x, dtype = np.float32)
# scale = StandardScaler()
# train_x = scale.fit_transform(train_x_ori)


file = open("data/new_matrix.pkl", "rb")

data = pickle.load(file)
mapping = data["mapping1"]
data = data["matrix"]

file.close()

# scale = StandardScaler()
# train_x = scale.fit_transform(data)
train_x = pca(data, 20)
scale = my_scaler()
data += 1.0
train_x =  scale.fit(data)



def plot_res(X = None, out = None, K = 3):
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'b', 'g', 'r']
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    X = train_x
    ax = plt.figure().subplots(1, 1)
    ax1 = ax
    for k, color in zip(range(K), colors):
        data = X[np.where(out == k)[0], :]
        print(data.shape[0])
        x, y = data[:, 0], data[:, 1]
        ax1.scatter(x, y, c=color)
    plt.show()




# t1 = kmeans(3, max_iter = 10000)
# out1 = t1.fit(train_x)
# plot_res(train_x, out1, K = 3)
# score1 = silhouette_score(train_x, out1)
# print(score1)

# t2 = KMedoids(3, max_iter = 10000)
# out2 = t2.fit(train_x)
# plot_res(train_x, out2, K = 3)
# score2 = silhouette_score(train_x, out2)
# print(score2)

# clu = 3
# t3 = Kmeanspp(k = clu, max_iter=10000)
# out3 = t3.fit(train_x).labels
# plot_res(train_x, out3, K = clu)
# score3 = silhouette_score(train_x, out3)
# print(score3)


t4 = agg_clustering()
out4 = t4.fit(train_x)
out4_ = t4.cluster_merge(train_x, out4, threshold = 0.1)
score4 = silhouette_score(train_x, out4)
score4_ = silhouette_score(train_x, out4_)
t4.plot_res(train_x, out4_)
print(np.unique(out4).shape[0])
print(np.unique(out4_).shape[0])
print(score4)
print(score4_)


# out1 = pd.DataFrame(out1)
# out1.to_csv("data/result.csv")
