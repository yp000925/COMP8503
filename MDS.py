''''
Code for assignment 1
Multidimensional Scaling method
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.datasets import make_swiss_roll

def cal_Gram(M):
    '''
    Get Gram matrix of input matrix
    :param M: input distance matrix
    :return: G: Gram matrix representation
    '''
    dist_i = M.mean(axis=1) # 行
    dist_j = M.mean(axis=0) # 列
    dist_avg = M.mean()
    # 初始化矩阵B
    G = np.empty(shape=M.shape, dtype=np.float64)
    # 得到B
    for i in range(len(G)):
        for j in range(len(G)):
            G[i, j] = -(M[i, j]**2 - dist_i[i]**2 - dist_j[j]**2 + dist_avg**2) / 2
    return G

# prepare the original 3-D dataset swiss roll and plot
from time import time
N = 2000
x,t = make_swiss_roll(N,noise=0.1,random_state=42)
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=t, cmap=plt.cm.Spectral)
ax.view_init(4, -72)
plt.savefig("Ori_fig.png")
# plt.show()


# dimension reduction to 2D data
from sklearn.metrics.pairwise import euclidean_distances
d =2
D = euclidean_distances(x,x)
t0 = time()
# get Gram matrix of original distance matrix
B = cal_Gram(D)
# SVD to get eigenvalue and eigenvectors of the Gram Matrix
val, vec = np.linalg.eig(B)
# calculate the transformed matrix by using the closed-formed equations
A = np.diag(val[np.argsort(val)[-1:-3:-1]].real)
A = np.sqrt(A)
V = vec[:,np.argsort(val)[-1:-3:-1]].real
y = np.dot(A, V.T).T

t1 = time()
# plot
print("mds: %.2g sec" % ( t1 - t0))
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
ax.scatter(y[:, 0], y[:, 1], c=t, cmap=plt.cm.Spectral)
ax.set_title("mds dimension reduction: %.2g sec" % ( t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis('tight')
plt.savefig("output.png")
# plt.show()


# from sklearn import  manifold
# t0 = time()
# mds = manifold.MDS(n_components=2, max_iter=1000,n_init=1)
# y=mds.fit_transform(x)
# t1 = time()