from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from scipy import stats
import numpy as np
import pandas as pd


def one_hot_encode(data):
    df_onehot = pd.get_dummies(data, columns=['Country', 'Item'], prefix=['Country', 'Item'])
    data = df_onehot.loc[:, df_onehot.columns != 'Yield (hg/ha)']
    data['Yield (hg/ha)'] = df_onehot['Yield (hg/ha)']
    return data

def feature_scaling(data):
    y = data['Yield (hg/ha)']
    X = data.drop('Yield (hg/ha)', axis=1)
    scaler = MinMaxScaler()
    data_without_yield = pd.DataFrame(scaler.fit_transform(X), index=y.index)
    data_without_yield.columns = X.columns
    data_without_yield.insert(len(data_without_yield.columns), 'Yield (hg/ha)', y)
    data = data_without_yield
    return data

def filter_data(data):
    y = data['Yield (hg/ha)']
    X = data.drop('Yield (hg/ha)', axis=1)
    z_scores = stats.zscore(X)
    abs_z_scores = np.abs(z_scores)

    filtered_entries = (abs_z_scores < 11).all(axis=1)
    X = X[filtered_entries]

    X.insert(len(X.columns), 'Yield (hg/ha)', y)
    filtered_data = X
    
    return filtered_data
