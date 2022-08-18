#### 2022/7/28

##### 内网ip地址分析

##### Class C

192.168.0.0\16 1581
192.168.102.0\24 23
192.168.12.0\24 2
192.168.151.0\24 11
192.168.42.0\24 54
192.168.220.0\24 45
192.168.99.0\24 1
192.168.142.0\24 48
192.168.211.0\24 16
192.168.121.0\24 50
192.168.72.0\24 95
192.168.130.0\24 78
192.168.122.0\24 50
192.168.150.0\24 15
192.168.205.0\24 4
192.168.127.0\24 51
192.168.195.0\24 3
192.168.128.0\24 3
192.168.190.0\24 19
192.168.55.0\24 1
192.168.10.0\24 133
192.168.202.0\24 9
192.168.32.0\24 122
192.168.171.0\24 14
192.168.192.0\24 1
192.168.191.0\24 1
192.168.60.0\24 118
192.168.90.0\24 117
192.168.81.0\24 73
192.168.51.0\24 4
192.168.41.0\24 46
192.168.131.0\24 35
192.168.180.0\24 106
192.168.31.0\24 5
192.168.125.0\24 68
192.168.50.0\24 9
192.168.29.0\24 19
192.168.112.0\24 13
192.168.120.0\24 20
192.168.52.0\24 40
192.168.110.0\24 30
192.168.40.0\24 29



##### ClassB

172.18.0.0\12 82
172.17.0.0\12 12
172.21.0.0\12 9
172.16.0.0\12 4
172.19.0.0\12 49
172.22.0.0\12 19



##### Class A

10.200.0.0\8 27
10.20.0.0\8 7
10.130.0.0\8 30
10.230.0.0\8 45
10.120.0.0\8 1



源ip地址中，内网ip个数：1866

与内网ip通信目的ip个数：22824

属于内网ip源地址目的地址交集个数：1622

属于内网ip间的通信instance个数：2070633



#### 2022/7/30：

- 提取出n个内网源ip
- 直接建立内网ip间的$n\times n$的流量矩阵
- 按照srcip-dstip-pair 得到按时间戳升序的数据集
- 流量聚合（0.5h的聚合窗口）
- 得到流量矩阵



#### 未完成：

- 流量大小的计算


大小：$total = Total.Length.of.Bwd.Packets + Total.Length.of.Fwd.Packets$

| flow_byts_s                 | 流字节率，即每秒传输的数据包字节数 |
| --------------------------- | ---------------------------------- |
| flow_duration               | 流持续时间                         |
| flow_pkts_s                 | 流包率，即每秒传输的数据包数       |
| Total.Length.of.Fwd.Packets | 正向数据包总大小（Bytes）          |
| Total.Length.of.Bwd.Packets | 反向数据包总大小（Bytes）          |

- 流量聚合


#### 已完成

##### 流量矩阵matrix.pkl

```python
dict = {"matrix" : traffic_matrix, "mapping" : reverse_map}
traffic_matirx : size = (1866, 1866)
reverse_map : map index to ip
```





#### 2022/7/31：

- 目的ip的选取：通信频次最高的m个
- 流量大小 $traffic = Total.Length.of.Fwd.Packets$，只选择发出的流量

现在的问题：

1. 两种情况下，流量矩阵中绝大多数内网ip都只和6个ip通信
2. 聚类结果分析，只采用了轮廓系数
3. K-medoids时间复杂度$O((kn)^2)$

```python
dict = {"matrix" : traffic_matrix, "mapping1" : reverse_map， "mapping2" : re_col_map}
traffic_matirx : size = (1866, 1000)
reverse_map : map index to ip
re_col_map : map index to dst_ip
```



#### 2022/8/1：

- Nsdi论文分成聚类后的聚簇合并算法实现



#### 问题汇总：

- 流量聚合
- 流量统计时，流量方向（考虑发往A以及从A收到的流量或者只考虑A发出的流量）。

##### Each traffic matrix in the dataset contains uniformly sampled VM-to-VM traffic aggregated over a 1-hour window。The $i, j-th$ entry $y_{ij}$ of $Y$ represents the volume of traffic (in bytes) from VM $i$ to VM $j$, where $i\in[n],\ j\in[m]$.（NSDI22）

- 流量采样

均匀采样

全采样

- 稀疏矩阵

降维

- 特征缩放

standardization

log scaling
$$
\large{x=\frac{x-\mu}{\sigma}}
\\\large{x=log(x)}
$$

- 距离度量

欧式距离（只考虑数值关系）：

```python
a = [1,1,1,0,0,0]
b = [1,1,1,2,2,2]
c = [3,3,3,0,0,0]
# b、c和a的欧氏距离一样，但b、c的通信模式不一样

# cosine
d(a, b) = 0.5527
d(a, c) = 0.0
```

余弦相似度：
$$
\large{cos\theta=\frac{\sum^{n}_{i}(X_i\times Y_i)}{\sqrt{\sum^{n}_{i}X_i^2}\times \sqrt{\sum^{n}_{i}Y_i^2}}}
\\
\large{dist = 1-cos\theta}
$$
- 聚类结果分析



#### 2022/8/1-2022/8/11

只选取1581个用户ip，属于192.168.0.0(16)子网，对（1581，1000）的流量矩阵进行聚类

结果

- K= 6, kmeans, silhouette_score = 0.27502584

![image-20220809170305353](C:\Users\LIlong\AppData\Roaming\Typora\typora-user-images\image-20220809170305353.png)

- K=6，kmedoids, silhouette_score = 0.19997384

![image-20220809170324266](C:\Users\LIlong\AppData\Roaming\Typora\typora-user-images\image-20220809170324266.png)

- K=6，kmeans++,  silhouette_score = 0.2771018

![image-20220809170355996](C:\Users\LIlong\AppData\Roaming\Typora\typora-user-images\image-20220809170355996.png)





轮廓系数(**ICA**)：

|      | kmeans   | kmedoids | kmeans++ |
| ---- | -------- | -------- | -------- |
| 3    | 0.356022 | 0.318479 | 0.356022 |
| 4    | 0.258677 | 0.29146  | 0.311548 |
| 5    | 0.297573 | 0.269274 | 0.316491 |
| 6    | 0.268114 | 0.2436   | 0.249101 |
| 7    | 0.257733 | 0.162759 | 0.279811 |
| 8    | 0.246162 | 0.253812 | 0.299934 |
| 9    | 0.270633 | 0.218649 | 0.282665 |
| 10   | 0.280007 | 0.285532 | 0.262109 |

#### 簇内分布

----kmeans----
[ 51. 174.  91.]
[556. 196. 309.]
[ 36. 105.  63.]
----kmedoids----
[553. 215. 327.]
[ 58. 113.  75.]
[ 32. 147.  61.]
----kmeans++----
[ 46. 104.  64.]
[547. 244. 314.]
[ 50. 127.  85.]



修改算法的cost计算方法：
$$
SSE = \Large{\sum_{k=1}^{K}\sum_{x\in Ci}dist(x, c_i)^2}
$$



#### 2022/8/12-2022/8/13

#### truncatedSVD(X, 10)

|      | kmeans   | kmedoids | kmeans++ |
| ---- | -------- | -------- | -------- |
| 3    | 0.404734 | 0.422628 | 0.431379 |
| 4    | 0.413069 | 0.458929 | 0.386368 |
| 5    | 0.429307 | 0.517802 | 0.430716 |
| 6    | 0.420463 | 0.488839 | 0.43877  |
| 7    | 0.432252 | 0.446098 | 0.481298 |
| 8    | 0.453908 | 0.459162 | 0.499994 |
| 9    | 0.42564  | 0.515065 | 0.447808 |
| 10   | 0.499036 | 0.464719 | 0.479969 |

#### pca(X, 0.90)

|      | kmeans   | kmedoids | kmeans++ |
| ---- | -------- | -------- | -------- |
| 3    | 0.546262 | 0.521168 | 0.518278 |
| 4    | 0.53438  | 0.52178  | 0.522266 |
| 5    | 0.513808 | 0.454851 | 0.529671 |
| 6    | 0.473919 | 0.472426 | 0.512674 |
| 7    | 0.455575 | 0.493788 | 0.460509 |
| 8    | 0.455324 | 0.498569 | 0.478141 |
| 9    | 0.500842 | 0.500485 | 0.504375 |
| 10   | 0.331584 | 0.503398 | 0.493125 |



Use the method from the paper to cut the ward tree.

#### From hierarchical to flat clustering

for a non-leaf node with height $h$, calculate the inconsistency:
$$
\large{H=\{h_0,h_1,...\}}
\\
\large{inc=\frac{h-\overline{H}}{\sigma}}
$$
the algorithm merges nested clusters when the inconsistency score is less than a threshold, $\mu$.

use elbow method to choose the threshold $\mu$.



#### Question:

1. How to get the traffic matrix 
2. how to determine the threshold in cluster merging
3. I find the result of inconsistency analysis conflicts with the paper.  ''Closer to the leaves of the dendrogram, inconsistency values will be small. They will increase at non-leaf nodes higher in the dendrogram.''  Some closer nodes has bigger inconsistency values than some higher nodes. A non-leaf node has an inconsistency value $inc$, its child node may have a value higher than $inc$.
4. cut the dendrogram using a height(distance) threshold.

#### github discussion：

1. Method of Parameter selection

cluster number(Kmeans, Kmeans++, Kmedoids)

2. Evaluation method of dimensionality reduction results

3. method to deal with sparse matrix

we find the matrix from the dataset is sparse

4. distance metrics used in the clustering algorithm

  euclidean

  cosine





#### a k-medoids method:

1. ##### Step 1: (Select initial medoids) 

1-1. Calculate the distance between every pair of all objects based on the chosen dissimilarity measure (Euclidean distance in our case). 

1-2. Calculate $v_j$ for object $j$ as follows: 
$$
v_j=\sum_{i=1}^{n}\frac{d_{ij}}{\sum_{l=1}^{n}d_{il}}, j=1,\ ...\ ,n
$$
1-3. Sort vj’s in ascending order. Select k objects having the first k smallest values as initial medoids. 

1-4. Obtain the initial cluster result by assigning each object to the nearest medoid. 

1-5. Calculate the sum of distances from all objects to their medoids. 

2. ##### Step 2: (Update medoids) 

Find a new medoid of each cluster, which is the object minimizing the total distance to other objects in its cluster. Update the current medoid in each cluster by replacing with the new medoid. 

3. #### Step 3: (Assign objects to medoids) 

3-1. Assign each object to the nearest medoid and obtain the cluster result. 

3-2. Calculate the sum of distance from all objects to their medoids. If the sum is equal to the previous one, then stop the algorithm. Otherwise, go back to the Step 2.
