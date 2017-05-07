__author__ = 'cissnei'

from matplotlib import pyplot as plt
import numpy as np
np.random.seed(4711) # for repeatability

c1 = np.random.multivariate_normal([10,0], [[3,1], [1,4]], size=[100,])

l1 = np.zeros(100)
l2 = np.ones(100)

c2 = np.random.multivariate_normal([0,10], [[3,1], [1,4]], size=[100,])

# add nosie:
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
