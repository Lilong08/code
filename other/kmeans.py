import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

def distance(flow, clusters):
    """
    Calculates the l2 distance between a flow and k clusters
    :param flow: array, (m, )
    :param clusters: numpy array of shape (k, m) where k is the # of clusters and m is the # fof dim
    :return: numpy array of shape(k, 0) where k is the # of clusters
    """
    
    distance =  np.linalg.norm(flow - clusters, axis = 1)
    return distance
def calc_distance(data, clusters):
    """Distance Matrix"""

    dist_mat = pairwise_distances(data, clusters)
    return dist_mat


def kmeans_run(flows, k, max_iter=None, dist=np.mean):
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

        distances = calc_distance(flows, clusters)
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            print("clusters no change")
            break

        ### 确定每个簇的新质心

        for cluster in range(k):
            clusters[cluster] = dist(flows[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters
        cnt += 1
        if (cnt > max_iter):
            print("max iteration")
            break

    return last_clusters

class kmeans:
    ''' kmeans for clustering '''
    '''
    k: cluster number
    max_iter: max iteration numbers
    '''

    def __init__(self, k, max_iter=1000):
        self.max_iter = max_iter
        self.k = k

    def fit(self, X):
        labels = kmeans_run(X, self.k, self.max_iter, dist = np.mean)
        return labels





if __name__ == '__main__':

    n_samples = 200
    centers = [[2, 2], [-1, -1], [1, -2]]
    X, labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1, random_state=0)

    trainer = kmeans(2, 100000)
    labels_ = trainer.fit(X)
    print(labels_)