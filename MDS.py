''''
Code for assignment 1
Multidimensional Scaling method
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.datasets import make_s_curve,make_swiss_roll

# prepare the original 3-D dataset swiss roll and plot
from time import time
N = 2000
x,t = make_swiss_roll(N,noise=0.1,random_state=42)
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=t, cmap=plt.cm.Spectral)
ax.view_init(4, -72)
plt.show()


# from sklearn import  manifold
# t0 = time()
# mds = manifold.MDS(n_components=2, max_iter=1000,n_init=1)
# y=mds.fit_transform(x)
# t1 = time()
# dimension reduction to 2D data
from sklearn.metrics.pairwise import euclidean_distances
d =2
D = euclidean_distances(x,x)
t0 = time()
# Get Gram matrix
dist_i = D.mean(axis=1) # 行
dist_j = D.mean(axis=0) # 列
dist_avg = D.mean()
# 初始化矩阵B
B = np.empty(shape=D.shape, dtype=np.float64)
# 得到B
for i in range(len(B)):
    for j in range(len(B)):
        B[i, j] = -(D[i, j]**2 - dist_i[i]**2 - dist_j[j]**2 + dist_avg**2) / 2
# 求B的特征值和特征向量
val, vec = np.linalg.eig(B)
# 中间矩阵
m1 = np.diag(val[np.argsort(val)[-1:-3:-1]].real)
m1 = np.sqrt(m1)
m2 = vec[:,np.argsort(val)[-1:-3:-1]].real
# 降维后的矩阵
y = np.dot(m1, m2.T).T
t1 = time()


print("mds: %.2g sec" % ( t1 - t0))
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
ax.scatter(y[:, 0], y[:, 1], c=t, cmap=plt.cm.Spectral)
ax.set_title("mds: %.2g sec" % ( t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis('tight')
plt.show()

