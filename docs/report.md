#### problem description

The problem is to identify similar network users, or to cluster the same/similar VMs in a Cloud scenario. We hope that VMs/users in a cluster are performimg the same/similar functions. 

#### existing studies

paper:  CloudCluster: Unearthing the Functional Structure of a Cloud Service 

the paper above uses the VM-to-VM traffic matrix to unearth the functional structure of a cloud service by clustering techniques. The proposed ClouCluster can determine that VMs in a cluster likely perform the same function. 

The scenario and purpose are similar to our work, so we think clustering the VM-to-VM traffic matrix also works in our problem. 

In the paper, they use Hierarchical Clustering to group VMs, as for data pre-processing, log scaling and SVD are used. 

#### current work

We have implemented four clustering algorithms, including kmeans\kmedoids\kmeas++\hierarchical clustering(with cluster merging). 

we investigated a dataset "IP Network Traffic Flows Labeled with 75 Apps", which is used for personalized service degradation policies. The dataset is also used to group users with similar consumption behavior. So, we just try to use the dataset to cluster users by traffic matrix.

About expeiment result:

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



- K= 6, kmeans

![image-20220809203417030](C:\Users\LIlong\AppData\Roaming\Typora\typora-user-images\image-20220809203417030.png)

- K=6，kmedoids

![image-20220809203440647](C:\Users\LIlong\AppData\Roaming\Typora\typora-user-images\image-20220809203440647.png)

- K=6，kmeans++

![image-20220809203506885](C:\Users\LIlong\AppData\Roaming\Typora\typora-user-images\image-20220809203506885.png)

We only got above coarse results. But the dataset is different from our data.

#### discussion

1. Our data hava no ground truth, how can we evaluate the final clustering results?

