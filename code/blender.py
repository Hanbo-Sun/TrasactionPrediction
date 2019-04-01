#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:09:16 2019

@author: hanbosun
"""
# %%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
sns.set()
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
os.chdir('/Users/hanbosun/Documents/GitHub/TrasactionPrediction/')

# %%
# inspired by [...]
s1=pd.read_csv('submission/lgb_submission_42.csv')['target']
s2=pd.read_csv('submission/lgb_submission_72.csv')['target']
s3=pd.read_csv('submission/lgb_submission_896.csv')['target']
s = pd.DataFrame({'s1': s1, 's2': s2, 's3': s3})


# %% Submissions analysis
# since we use AUC, and distribution of the probability is not Normal, Kendal correlation is more appropriate
kendall = s.corr(method = 'kendall') # spearman pearson
sns.heatmap(kendall, annot = True, fmt = ".3f")


# %% Submissions
# Simple blender
submission = pd.read_csv('input/sample_submission.csv')

submission['target'] = 0.7*s1 + 0.3*s2
filename="submission/blended_submission_{:%Y-%m-%d_%H_%M}.csv".format(datetime.now())
submission.to_csv(filename, index=False)
