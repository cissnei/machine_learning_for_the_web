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
df=df.fillna(-1)