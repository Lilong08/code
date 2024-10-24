import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
# 读取CSV文件
def read_csv(file_path):
    data = pd.read_csv(file_path, header=0, index_col=0)
    data = data.iloc[:, 1:1000]
    return data

# 对数据进行标准化处理（可选，取决于数据的量纲差异）
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# K-Means 聚类
def perform_kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_, kmeans

# 绘制聚类结果图
def plot_clusters(data, labels, n_clusters):
    plt.figure(figsize=(8, 6))
    
    # 根据聚类数量生成不同的颜色
    colors = plt.cm.get_cmap('tab10', n_clusters)
    
    # 绘制每个簇
    for cluster in range(n_clusters):
        cluster_data = data[labels == cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                    label=f'Cluster {cluster}', color=colors(cluster))
    
    plt.title(f'K-Means Clustering with {n_clusters} Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    # plt.show()
    plt.savefig('new-data-2-1000/wbq-test-' + str(n_clusters) + '.png')

# 主函数
def main():
    
    input_csv = 'data-2/new_data2.csv'  # 输入文件路径
    # 读取数据
    data = read_csv(input_csv)
    
    # 预处理数据（移除非数值列，如果有）
    data_numeric = data.select_dtypes(include=['number'])
    
    # 对数据进行标准化处理
    scaled_data = preprocess_data(data_numeric)
    score = []
    
    for n_clusters in range(2, 11):
        # 聚类
        labels, kmeans = perform_kmeans_clustering(scaled_data, n_clusters)
        
        # 输出聚类结果
        print("聚类标签：", labels)
        
        score1 = silhouette_score(scaled_data, labels)
        score.append(score1)
        print("分数：", score1)
        
        # 如果数据维度大于2，只绘制前两列的特征
        plot_data = scaled_data[:, :2] if scaled_data.shape[1] > 2 else scaled_data
        
        # 绘制聚类结果
        plot_clusters(plot_data, labels, n_clusters)

    print(score)

if __name__ == '__main__':
    main()