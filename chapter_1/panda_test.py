__author__ = 'cissnei'

import pandas as pd
import numpy as np
import random

data = pd.read_csv("ad-dataset/ad.data", header=None, low_memory=False)

adindices = data[data.columns[-1]]=='ad.'
data.loc[adindices,data.columns[-1]]=1
noadindices = data[data.columns[-1]]=='nonad.'
data.loc[noadindices,data.columns[-1]]=0

data[data.columns[-1]] = data[data.columns[-1]].astype(float)
data=data.replace({'?':np.nan})
data=data.replace({' ?':np.nan})
data=data.replace({'  ?':np.nan})
data=data.replace({'   ?':np.nan})
data=data.replace({'    ?':np.nan})
data=data.replace({'     ?':np.nan})

data = data.fillna(-1)
data = data.apply(lambda x: pd.to_numeric(x))

dataset = data.values[:,:]
np.random.shuffle(dataset)
df = dataset[:,:-1]
labels = dataset[:,-1].astype(float)
ntrainrows = int(len(df)*.8)

train = df[:ntrainrows:]
trainlabels = labels[:ntrainrows]
test = df[ntrainrows:,:]
testlabels = labels[ntrainrows:]

from sklearn.svm import SVC
clf = SVC(gamma=0.001, C=100.)
clf.fit(train, trainlabels)

score = clf.score(test, testlabels)
print 'score: :', score

