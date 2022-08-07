from cProfile import label
from re import S
from time import sleep
from tkinter import N
import numpy as np
from sklearn.metrics import pairwise_distances
from itertools import cycle
from typing import List, Dict, Set, Any



# a = np.array([1,2,3,3,3,31,2,1])
# out = np.array([0,0,1,1,1,2,0])
# c = np.where(out == 0)[0]
# data = np.random.randint(1, 10, (2,3))
# print(np.sum(data,axis=1))

# prob = a/np.sum(a)

# d = np.random.choice(a,2, p = prob)
# print(d)




# aa = [1,2,3,4]
# bb = [1,2,3,4,4]
# cc = [1,23,4,4]

# for _ in zip(aa, bb, cc):
#     print(_)

# d = np.random.randint(1,10, (2, 3))
# print(d)
# dd = np.random.randint(1, 10, (5, 3))
# dist = pairwise_distances(d, dd)**2
# print(dist.shape)
# print(dist)


# tmp = np.random.randint(1, 10, (5, 5))
# dist = pairwise_distances(tmp, metric = 'cosine')
# for i in range(dist.shape[0]):
#     dist[i][i] = np.inf

# print(dist)
# value_min = np.min(dist, axis = 1)
# idx_min = np.argmin(dist, axis = 1)

# print(value_min)
# print(idx_min)

# a = np.array([[10,22,3,],[2,3,4],[3,4,5]])

# value_min = np.min(a, axis = 1)
# idx_min = np.argmin(a, axis = 1)
# x = np.argmin(value_min)
# y = idx_min[x]
# print(x, y)


# c = [1,2,1]
# d = [2,4,5]
# c = np.array(c)
# d = np.array(d)

# import matplotlib.pyplot as plt
# plt.figure(1)

# plt.plot(c, d, 'r.')
# plt.show()



# class A:
#     def __init__(self) -> None:
#         self.a = 1
#         self.b = 2
#     def __getattr__(self, __name):
#         print("%s is not callable!"%__name)
#     def __getattribute__(self, __name: str) -> Any:
#         print("%s is called!"%__name)
        
#     def func(self) -> Any:
#         print("test")
        
# obj = A()
# print(obj.bb)
# print(obj.a)

  
class dataset:
    def __init__(self, file_path):
        X = np.random.randint(1, 100, (14,4))
        y = np.random.randint(1, 100, (14,))
        self.train = X
        self.label = y
        self.idx = 0
    
    def __len__(self):
        assert len(self.train) == len(self.label)
        return len(self.train)

    def __getitem__(self, key):
        assert key < len(self)
        return self.train[key], self.label[key]

    def __next__(self):
        assert self.idx < len(self)
        tmp = self.label[self.idx]
        self.idx += 1
        return tmp
    # def __iter__(self):
    #     l = len(self)

    #     x = []
    #     y = []
    #     for i in range(l//self.batch_size):
    #         x.append(self.train[i * self.batch_size: (i + 1) * self.batch_size])
    #         y.append(self.label[i * self.batch_size: (i + 1) * self.batch_size])
        
    #     if(l % self.batch_size != 0):
    #         s = l // self.batch_size * self.batch_size
    #         x.append(self.train[s:])
    #         y.append(self.label[s:])

    #     x = np.array(x, dtype = object)
    #     y = np.array(y, dtype = object)
    #     return zip(x, y)




# ds = dataset("")
# print(ds.label)
# for i in range(15):
#     print(next(ds))

# def sq():
#     for x in range(4):
#         c = yield x**2
#     print(type(c))
# dl = dataset("")


# print(dl)

# b = sq()
# print(b)
# for i in b:
#     print(i)

# l = [1,2,3]

# ll = iter(l)
# for i in range(3):
#     print(next(ll))

# def fun(n : int):
#     for i in range(n):
#         yield (i + 2, "ss")
#     print("sfsd")

# f = fun(10)
# for i in f:
#     print(i)


# a = np.array([[0,0,0],[2,2,4],[2,3,1]])

# dist = pairwise_distances(a, metric='cosine')
# print(np.argmin(dist, axis = 1))
# print(dist)

x = np.linspace(-10, 10, 100)
y = 1/(1 + np.e**(-x))

import matplotlib.pyplot as plt


ax = plt.figure().subplots(1, 1)

ax.scatter(x, y)
plt.show()