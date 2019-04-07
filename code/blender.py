#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:09:16 2019

@author: hanbosun
"""
# %%




# %% simple blender

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

s42=pd.read_csv('submission/lgb_submission_42.csv')['target']
s72=pd.read_csv('submission/lgb_submission_72.csv')['target']

s0=pd.read_csv('submission/lgb_submission_seed0.csv')['target']
s1=pd.read_csv('submission/lgb_submission_seed1.csv')['target']
s2=pd.read_csv('submission/lgb_submission_seed2.csv')['target']
s3=pd.read_csv('submission/lgb_submission_seed3.csv')['target']
s4=pd.read_csv('submission/lgb_submission_seed4.csv')['target']
s5=pd.read_csv('submission/lgb_submission_seed5.csv')['target']
s6=pd.read_csv('submission/lgb_submission_seed6.csv')['target']
s7=pd.read_csv('submission/lgb_submission_seed7.csv')['target']
s12=pd.read_csv('submission/lgb_submission_seed12.csv')['target']
s13=pd.read_csv('submission/lgb_submission_seed13.csv')['target']
s14=pd.read_csv('submission/lgb_submission_seed14.csv')['target']
s15=pd.read_csv('submission/lgb_submission_seed15.csv')['target']
s16=pd.read_csv('submission/lgb_submission_seed16.csv')['target']
s17=pd.read_csv('submission/lgb_submission_seed17.csv')['target']
s18=pd.read_csv('submission/lgb_submission_seed18.csv')['target']
s19=pd.read_csv('submission/lgb_submission_seed19.csv')['target']

s = pd.DataFrame({'s0': s0, 's1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5, 's6': s6, 's7': s7,
                   's12': s12, 's13': s13, 's14': s14, 's15': s15, 's16': s16, 's17': s17, 's18': s18, 's19': s19})

# since we use AUC, and distribution of the probability is not Normal, Kendal correlation is more appropriate
kendall = s.corr(method = 'kendall') # spearman pearson


submission = pd.read_csv('input/sample_submission.csv')

submission['target'] = (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 +    s12 + s13 + s14 + s15 + s16 + s17 + s18 + s19) /16
filename="submission/blended_submission_{:%Y-%m-%d_%H_%M}.csv".format(datetime.now())
submission.to_csv(filename, index=False)


# %% rank blender: https://www.kaggle.com/roydatascience/blender-of-0-901-solutions

import numpy as np
import pandas as pd
import os

from scipy.stats import rankdata
LABELS = ["target"]
predict_list = []

predict_list.append(pd.read_csv('submission/lgb_submission_42.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_72.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed0.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed1.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed2.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed3.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed4.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed5.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed6.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed7.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed12.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed13.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed14.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed15.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed16.csv')[LABELS].values)
predict_list.append(pd.read_csv('submission/lgb_submission_seed17.csv')[LABELS].values)

print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(1):
        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  
predictions /= len(predict_list)


submission = pd.read_csv('input/sample_submission.csv')
submission[LABELS] = predictions
filename="submission/super_blend_{:%Y-%m-%d_%H_%M}.csv".format(datetime.now())
submission.to_csv(filename, index=False)


#%% PCA blender: https://www.kaggle.com/darbin/pca-blender-of-0-901-solutions
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from datetime import datetime
import os
print(os.listdir("../input"))

# inspired by [...]
s1=pd.read_csv('../input/s1v8-nanashi-90-lines-solution-0901-fast/s1-v8.csv')['target']#[1]
s2=pd.read_csv('../input/santander-magic-lgb-0-901/submission.csv')['target']#[2]
s3=pd.read_csv('../input/s3v32-ashish-gupta-eda-pca-scaler-lgbm/s3-v32.csv')['target']#[3]
s4=pd.read_csv('../input/eda-pca-simple-lgbm-on-kfold-technique/submission26.csv')['target']#[4]
s5=pd.read_csv('../input/lgb-2-leaves-augment/lgb_submission.csv')['target']#[5]
s6=pd.read_csv('../input/s6v1-ole-morten-light-gbm-with-data-augment/s6-v1.csv')['target']#[6]
s7=pd.read_csv('../input/s7v19-subham-sharma-what-is-next-in-santander/s7-v19.csv')['target']#[7]
s8=pd.read_csv('../input/s8v5-joshua-reed-santander-customer-transaction/s8-v5.csv')['target']#[8]
s9=pd.read_csv('../input/s9v16-gauravtambi-lgbm-augmentation/s9-v16.csv')['target']#[9]
s10=pd.read_csv('../input/best-parameters-lb-0-900/submission.csv')['target']#[10]

submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')

solutions_set = pd.DataFrame({'s1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5, 's6': s6,
                              's7': s7, 's8': s8, 's9': s9, 's10': s10})
    

# since we use AUC, and distribution of the probability is not Normal, Kendal correlation is more appropriate
kendall = solutions_set.corr(method = 'kendall')
plt.figure(figsize=(10, 6))
plt.title('Kendall correlations between the 0.901-solutions')
sns.heatmap(kendall, annot = True, fmt = ".3f")


# Pearson correlations between the submissions
pearson = solutions_set.corr(method = 'pearson')
plt.figure(figsize=(10, 6))
plt.title('Pearson correlations between the 0.901-solutions')
sns.heatmap(pearson, annot=True, fmt=".3f")

# Density of the solutions
plt.figure(figsize=(10, 6))
plt.title('Density of the 0.901-solutions')
sns.kdeplot(s1, label = 's1', shade = True)
sns.kdeplot(s2, label = 's2', shade = True)
sns.kdeplot(s3, label = 's3', shade = True)
sns.kdeplot(s4, label = 's4', shade = True)
sns.kdeplot(s5, label = 's5', shade = True)
sns.kdeplot(s6, label = 's6', shade = True)
sns.kdeplot(s7, label = 's7', shade = True)
sns.kdeplot(s8, label = 's8', shade = True)
sns.kdeplot(s9, label = 's9', shade = True)
sns.kdeplot(s10, label = 's10', shade = True)

# Preprocessing: scaling (scale all the submissions to vars with mean = 0 and std = 1 since PCA doesn't like inputs with different scales)
scaler = StandardScaler()
solutions_set_scaled = pd.DataFrame(scaler.fit_transform(solutions_set),
                                    columns = ['s1', 's2', 's3', 's4', 's5', 's6', 
                                               's7', 's8', 's9', 's10'])
    
solutions_set_scaled.describe().applymap('{:,.2f}'.format)


# increase the weight for the s5
solutions_set_scaled['s5'] = solutions_set_scaled['s5'] * 3

# table style (red color)
def red_color(val):
    color = 'red'
    return 'color: %s' % color

# table: implement styles
solutions_set_scaled_style1 = solutions_set_scaled.describe().applymap('{:,.2f}'.format)
solutions_set_scaled_style1.style.applymap(red_color, subset = ['s5'])

# PCA
pca = PCA(n_components = 1)
factor = pca.fit_transform(solutions_set_scaled)
print(pca.explained_variance_ratio_)

# Loadings. These loadings can be interpreted as weights of each submission
pca.components_

# Pearson correlation of the submissions with the extracted factor
plt.figure(figsize=(10, 6))
plt.title('Pearson correlation of the extracted factor with the 0.901-solutions')
solutions_set_scaled['factor'] = factor
pearson_pca = solutions_set_scaled.corr(method = 'pearson')
sns.heatmap(pearson_pca, annot = True, fmt = ".3f")

# Kendall correlation of the submissions with the extracted factor
plt.figure(figsize=(10, 6))
plt.title('Kendall correlation of the extracted factor with the 0.901-solutions')
kendall_pca = solutions_set_scaled.corr(method = 'kendall')
sns.heatmap(kendall_pca, annot = True, fmt = ".3f")

# PCA blender
# Since we use AUC the values for the target don't really matter
# What matters is the order of the target values
submission['target'] = solutions_set_scaled['factor']

filename="blended_submission_{:%Y-%m-%d_%H_%M}.csv".format(datetime.now())
submission.to_csv(filename, index=False)

