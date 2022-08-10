# Dimensionality Reduction

#### linear techniques

the new variable being a linear combination of the original variables:
$$
s_i = w_{i, 1}x_1+w_{i,2}x_2+...+w_{i,p}x_p,\ for\ i = 1,...,k\\
x=Wx, W_{k\times p}
$$

- PCA

explained_variance_ratio_：计算每个特征的方差贡献率

- FA(Factor analysis)





#### ICA：

提取相互独立的属性，处理非高斯数据，各变量间相互独立

ICA data pre-process with PCA

#### t-SNE:

降维和可视化，不用于数据转换，维度过高不直接使用，非线性降维

#### NMF：

非负矩阵分解，给定非负矩阵$V$，NMF能够找到非负矩阵$W$和$H$，两个矩阵的乘积近似$V$

$W$矩阵：基础图像矩阵，相当于特征

$H$矩阵：系数矩阵
$$
\large{V_{n*m}=W_{n*k}*H_{k*m}}
$$

#### TruncatedSVD：

it can work with sparse matrices efficiently

it dose not center the data before computing the singular value decomposition