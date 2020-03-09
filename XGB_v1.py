# XGB_v1.py
# Created by KAC on 03/08/2020

""" This code will build a XGBoost model using RFE, cross validation, and parameter optimization.
Scoring methods is log loss."""

# Load packages
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.feature_selection import RFECV
from sklearn.metrics import log_loss
from xgboost import XGBRegressor as XGB
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer
from XGB_ModelBuilder import XGB_ModelBuilder

data_dir = '/Users/Koby/PycharmProjects/NCAA_2020/'
df_raw = pd.read_csv(data_dir + 'all_input.csv')

# Separate Winning and Losing Teams
df_win = pd.DataFrame()
df_win = df_raw.copy()
df_win['Result'] = 0

df_lose = df_win.copy()
df_lose = df_lose.multiply(-1)
df_lose['Result'] = 1
df_temp = pd.concat((df_win, df_lose))

# Create training data
X_train = df_temp.loc[df_win['Season'] < 2019]
X_test = df_temp.loc[df_win['Season'] == 2019]
X_train.drop(labels=['Season', 'WTeamID', 'LTeamID'], inplace=True, axis=1)
X_test.drop(labels=['Season', 'WTeamID', 'LTeamID'], inplace=True, axis=1)

y_train = X_train.Result.values
y_test = pd.DataFrame(X_test.Result.values)

X_train.drop(labels=['Result'], inplace=True, axis=1)
X_test.drop(labels=['Result'], inplace=True, axis=1)

seed = 7
X_train, y_train = shuffle(X_train, y_train, random_state=seed)
X_test, y_test = shuffle(X_test, y_test, random_state=seed)

predictions, preds_final = XGB_ModelBuilder(X_train, y_train, X_test, y_test)

for i in [0, 0.1, 0.2, 0.3, 0.4]:
    for j in [0, 0.1, 0.2, 0.3, 0.4]:
        for k in [0.35, 0.3, 0.25]:
            best_score = 1.0
            predictions.mask(predictions.between(i, j), other=k, inplace=True)
            pred_scr = round(log_loss(y_test, predictions), 5)
            if pred_scr < best_score:
                best_score = pred_scr
                i_best = i
                j_best = j
                k_best = k
                print(i,j,k, "2019 Score: ", pred_scr)

predictions.mask(predictions.between(i_best, j_best), other=k_best, inplace=True)

#TODO: Something liek this:
# clipped_preds = np.clip(preds, 0.05, 0.95)
# df_sample_sub.Pred = clipped_preds
# df_sample_sub.to_csv('2019_xgboost-repeatable.csv', index=False)

#TODO: The rest.
x=1

