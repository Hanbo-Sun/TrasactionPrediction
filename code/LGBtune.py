#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:47:16 2019

@author: hanbosun
"""

# -*- coding: utf-8 -*-

# %%

# LightGBM install: use conda: https://anaconda.org/conda-forge/lightgbm
# StratifiedKFold: This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
# KFold: Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).

# %%
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime

#user defined
import augment as ag
os.chdir('/Users/hanbosun/Documents/GitHub/TrasactionPrediction/')
random_state = 0


# %%
df_train = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/test.csv')
df_train_all = df_train

#%%
ids = np.arange(df_train_all.shape[0])
np.random.seed(random_state)

np.random.shuffle(ids)
#df_train = df_train_all.iloc[ids[:10000],:]
df_train = df_train_all.iloc[ids,:]
# %%
lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 16, # 13
    "learning_rate" : 0.005, # 0.01 0.002
    "bagging_freq": 5,
    "bagging_fraction" : 0.2, #0.4 0.335
    "feature_fraction" : 0.07, #0.05 0.5 0.1 (~sqrt p) 0.041
    "min_data_in_leaf": 80, #80 120
    "min_sum_hessian_in_leaf": 10, #10 100 1
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "seed": random_state,
    "verbosity": -1 # "num_threads":8 (default all)
}
   


# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
oof = df_train[['ID_code', 'target']]
oof['predict'] = 0
predictions = df_test[['ID_code']]
val_aucs = []

features = [col for col in df_train.columns if col not in ['target', 'ID_code']]
X_test = df_test[features].values

# %% change: down learning_rate, up early_stopping_rounds up N (# dataAugment); replicable seed
for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
    X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']
    X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']
    
    N = 10 # 5
    p_valid,yp = 0, 0
    for i in range(N):
        X_t, y_t = ag.augment_fast2(X_train.values, y_train.values, t=5, seed=random_state+i)
        #X_t, y_t =X_train.values, y_train.values
        print(X_t.shape)
    
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')
    
        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        
        lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000, 
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=5000, #3000
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(X_test)
    
    ##oof['predict'].iloc[val_idx] = p_valid/N
    #oof['predict'][val_idx] = p_valid/N
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
    
    predictions['fold{}'.format(fold+1)] = yp/N
    
# %% submission
mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))

predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)

sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = predictions['target']
filename="tune_submission_{:%Y-%m-%d_%H_%M}.csv".format(datetime.now())
sub_df.to_csv(filename, index=False)
#oof.to_csv('lgb_oof.csv', index=False)

    

#%%
