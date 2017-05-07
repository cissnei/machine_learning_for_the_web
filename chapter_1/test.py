__author__ = 'cissnei'

import time
import numpy as np

def sum_trad():
        start=time.time()
        X=range(10000000)
        Y=range(10000000)
        Z=[]
        for i in range(len(X)):
            Z.append(X[i] + Y[i])
        return time.time() - start

def sum_numPy():
        start = time.time()
        X=np.arange(10000000)
        Y=np.arange(10000000)
        Z=X+Y
        return time.time() - start

print 'trad: ', sum_trad()
print 'numPy: ', sum_numPy()
