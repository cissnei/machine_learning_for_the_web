__author__ = 'cissnei'

import pandas as pd
import numpy as np

df = pd.read_csv('ad-dataset/ad.data',header=None, low_memory=False)
df=df.replace({'?':np.nan})
df=df.replace({' ?':np.nan})
df=df.replace({'  ?':np.nan})
df=df.replace({'   ?':np.nan})
df=df.replace({'    ?':np.nan})
df=df.replace({'     ?':np.nan})
df=df.replace({'      ?':np.nan})
df=df.fillna(-1)

adindices = df[df.columns[-1]] =='ad.'
df.loc[adindices, df.columns[-1]]=1
nonadindices = df[df.columns[-1]] =='nonad.'
df.loc[nonadindices, df.columns[-1]]=0
df[df.columns[-1]] = df[df.columns[-1]].astype(float)
df.apply(lambda x: pd.to_numeric(x))

dataset = df.values[:,:]

np.random.shuffle(dataset)
data= dataset[:,:-1]
labels = dataset[:,:-1].astype(float)
ntrainrows=int(len(data)*.8)

train = data[:ntrainrows,:]
trainlabels = labels[:ntrainrows]
test = data[ntrainrows:,:]
testlabels = labels[ntrainrows:]

from sklearn.svm import SVC
clf = SVC(gamma=0.001, C=100.)
clf.fit(train, trainlabels)

score = clf.score(test, testlabels)
print 'score:',score