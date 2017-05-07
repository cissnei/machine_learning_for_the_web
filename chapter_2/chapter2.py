__author__ = 'cissnei'

from matplotlib import pyplot as plt
import numpy as np
from sklearn import mixture
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift

np.random.seed(4711) # for repeatability

c1 = np.random.multivariate_normal([10,0], [[3,1], [1,4]], size=[100,])

l1 = np.zeros(100)
l2 = np.ones(100)

c2 = np.random.multivariate_normal([0,10], [[3,1], [1,4]], size=[100,])

# add noise:
np.random.seed(1) # for repeatability
noise1x = np.random.normal(0,2,100)

noisely = np.random.normal(0,8,100)
noise2 = np.random.normal(0,8,100)

c1[:,0] += noise1x
c1[:,1] += noisely
c2[:,1] += noise2

fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(111)
ax.set_xlabel('x', fontsize=30)
ax.set_ylabel('y', fontsize=30)
fig.suptitle('classess', fontsize=30)

labels = np.concatenate((l1,l2),)
X = np.concatenate((c1,c2),)

pp1 = ax.scatter(c1[:,0], c1[:,1], cmap='prism', s=50, color='r')
pp2 = ax.scatter(c2[:,0], c2[:,1], cmap='prism', s=50, color='g')
ax.legend((pp1,pp2), ('class1', 'class2'), fontsize=35)
fig.savefig('class_normal_scatter.png')

fig.clf() # reset plt
fig, ((axis1, axis2), (axis3, axis4)) = plt.subplots(2,2,sharex='col', sharey='row')

# K-Means
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
pred_kmeans = kmeans.labels_
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='prism') # plot
# points with cluster dependent colors
axis1.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='prism')
axis1.set_ylabel('y', fontsize=40)
axis1.set_title('k-means', fontsize=20)

# Mean-Shift
ms = MeanShift(bandwidth=7)
ms.fit(X)
pred_ms = ms.labels_
axis2.scatter(X[:,0], X[:,1], c=pred_ms, cmap='prism')
axis2.set_title('mean-shift', fontsize=20)

# Gausian mixture
g = mixture.GMM(n_components=2)
g.fit(X)
pred_gmm = g.predict(X)
axis3.scatter(X[:,0], X[:,1], c=pred_gmm, cmap='prism')
axis3.set_xlabel('x', fontsize=40)
axis3.set_ylabel('y', fontsize=40)
axis3.set_title('Gaussian mixture', fontsize=20)

# Hierachical
# generate the linkage matrix
Z = linkage(X, 'ward')
max_d = 110
pred_h = fcluster(Z, max_d, criterion='distance')
axis4.scatter(X[:,0], X[:,1], c=pred_h, cmap='prism')
axis4.set_xlabel('x', fontsize=40)
axis4.set_title('hierarchical ward', fontsize=40)
fig.set_size_inches(18.5, 10.5)
fig.savefig('comp_clustering2.png', dpi=100)