
#%%
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
import os
os.chdir('C:/Users/sunyichi/Documents/GitHub/TrasactionPrediction')



# %%
def train_pred_xgb(model_params, train_x, train_y, valid_x, valid_y, test_x=None, verbose=True):
    train_data = xgb.DMatrix(data=train_x, label=train_y)
    valid_data = xgb.DMatrix(data=valid_x, label=valid_y)

    watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

    tr_model = xgb.train(
        dtrain=train_data,
        num_boost_round=100000,
        evals=watchlist,
        early_stopping_rounds=4000,
        verbose_eval=2000,
        params=model_params)

    y_valid_pred = tr_model.predict(xgb.DMatrix(valid_x), ntree_limit=tr_model.best_ntree_limit)

    if verbose:
        model_score = roc_auc_score(valid_y, y_valid_pred)
        print('XGB ROC AUC: {:07.6f}'.format(round(model_score, 6)))

    if test_x is not None:
        y_test_pred = tr_model.predict(xgb.DMatrix(test_x), ntree_limit=tr_model.best_ntree_limit)
        return y_valid_pred, y_test_pred
    else:
        return y_valid_pred


#%%
NUMBER_OF_FOLDS = 20
random_state = 0

#%%
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
train_all = train

#%%
ids = np.arange(train_all.shape[0])
np.random.seed(random_state)

np.random.shuffle(ids)
df_train = train_all.iloc[ids[:10000],:]
#df_train = train_all.iloc[ids,:]


#%%
y = train.target.reset_index(drop=True).values
x = train.drop(['ID_code', 'target'], axis=1).values.astype('float64')
x_test = test.drop(['ID_code'], axis=1).values.astype('float64')

ss = StandardScaler()
x = ss.fit_transform(x)
x_test = ss.transform(x_test)

qt = QuantileTransformer(output_distribution='normal')
x = qt.fit_transform(x)
x_test = qt.transform(x_test)
spw = float(len(y[y == 1])) / float(len(y[y == 0]))
print('scale_pos_weight: ' + str(spw))

#%%
params_xgb = {
    'eta': 0.02,
    'max_depth': 1,
    'subsample': 0.29,
    'colsample_bytree': 0.04,
    'lambda': 0.57,
    'alpha': 0.08,
    'min_child_weight': 5.45,
    'max_delta_step': 1.53,
    'scale_pos_weight': spw,
    'tree_method': 'gpu_hist', #'gpu_hist', 'hist'
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_gpus': 1,
    'verbosity': 0,
    'silent': True
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
        X_t, y_t = augment_fast2(X_train.values, y_train.values, t=5, seed=random_state+i)
        #X_t, y_t =X_train.values, y_train.values
        print(X_t.shape)
    
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')
    
        evals_result = {}
        pred_val_y, pred_test_y = train_pred_xgb(params_xgb, X_t, y_t, X_valid, y_valid, X_test)
        p_valid += pred_val_y
        yp += pred_test_y
    
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



