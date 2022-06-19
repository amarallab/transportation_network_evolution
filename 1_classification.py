import sys
import random
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scipy.stats as ss

from datetime import date
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,balanced_accuracy_score, mean_squared_error,r2_score,mean_absolute_error

sys.append('src')
from plot_style import *

def get_edge_slice(data, f_train_e=0.7, seed=30):
    df = data
    edges = set(df.Edge.unique())
    random.seed(seed)
    edge_train = set(random.sample(edges, int(f_train_e*len(edges))))
    edge_test = set([e for e in edges if e not in edge_train])
    df_se = df.loc[df['Edge'].isin(edge_train)].drop(columns = ['Edge'])
    df_de = df.loc[df['Edge'].isin(edge_test)].drop(columns = ['Edge'])
    return(df_se, df_de)

def df_to_XY(df, features, target='Removed'):
    if 'Year' in df.columns:
        df = df.drop(columns = ['Year'])
    if "Edge" in df.columns:
        df = df.drop(columns = ['Edge'])
    X = df.loc[:, features].to_numpy()
    y = df.loc[:, df.columns == target].to_numpy()
    return(X, y)

def simultaneous_test(df_se, df_de, features, best_params, save = True, name = None):
    if name is None:
        name = ''.join([w[0] for w in features]) + '_simultaneous'
    else:
        name =  name + '_simultaneous'
    year_list = list(df_se.Year.unique())
    res_df_de = df_de.copy()
    res_df_de['simultaneous_pred']= np.nan
    res_df_de['simultaneous_null']= np.nan
    for year in tqdm(year_list):
        X_train,y_train = df_to_XY(df_se[df_se.Year==year],features)
        ros = RandomUnderSampler()
        X_train,y_train = ros.fit_resample(X_train,y_train)
        X_test,y_test = df_to_XY(df_de[df_de.Year==year],features)
        y_train_null = y_train.copy()
        np.random.shuffle(y_train_null)
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        model_null = XGBClassifier(**best_params)
        model_null.fit(X_train, y_train_null)
        y_pred = model.predict(X_test)
        y_pred_null = model_null.predict(X_test)
        res_df_de.loc[res_df_de.Year==year, 'simultaneous_pred'] = y_pred
        res_df_de.loc[res_df_de.Year==year, 'simultaneous_null'] = y_pred_null
    if save:
        res_df_de.to_csv('./results/'+name+'.csv')
    return res_df_de

def nonsimultaneous_test(df_train, df_test, features, best_params, save=True, name = None):
    if name is None:
        name =  ''.join([w[0] for w in features]) + '_nonsimultaneous'
    else:
        name = name + '_nonsimultaneous'
    year_list = list(df_test.Year.unique())
    preds = []
    for year_train in tqdm(year_list):
        X_train,y_train = df_to_XY(df_train[df_train.Year==year_train],features)
        ros = RandomUnderSampler()
        X_train,y_train = ros.fit_resample(X_train,y_train)
        y_train_null = y_train.copy()
        np.random.shuffle(y_train_null)
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        model_null = XGBClassifier(**best_params)
        model_null.fit(X_train, y_train_null)
        i = year_list.index(year_train)
        for year_test in year_list[i:]:
            X_test, y_test = df_to_XY(df_test[df_test.Year==year_test],features)
            y_pred = model.predict(X_test)
            y_null = model_null.predict(X_test)
            preds.append([year_train ,year_test, y_test,y_pred,y_null])
    if save:
        import pickle
        with open('./results/'+ name +'.pkl', 'wb') as f:
            pickle.dump(preds, f)
    return preds

def all_shap_values(df1, df2, features, best_params, save=True, name = None):
    import shap
    if name is None:
        name =  ''.join([w[0] for w in features]) + '_SHAP'
    else:
        name =  name + '_SHAP'
    def get_temporal_order(shap_list):
        importance_array = []
        for shap_values in shap_list:
            array = -np.abs(shap_values).mean(axis=0)
            ranks = ss.rankdata(array)
            importance_array.append(ranks)
        return(np.array(importance_array))

    shap_values_list = []
    test_list = []
    year_list = []
    for i in tqdm(df2.Year.unique()):
        X_train,y_train = df_to_XY(df1[ df1.Year == i ].drop(columns = ['Year']),features)
        ros = RandomUnderSampler()
        X_train,y_train = ros.fit_resample(X_train,y_train)
        X_test,y_test = df_to_XY(df2[ df2.Year == i ].drop(columns = ['Year']), features)
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        year_list.append(i)
        test_list.append(pd.DataFrame(X_test,columns=features))
        shap_values_list.append(shap_values)
    if save:
        import pickle
        with open('./results/'+name+'.pkl', 'wb') as f:
            pickle.dump((test_list, year_list, shap_values_list), f)
    return (test_list, year_list, shap_values_list)

def BTF(train, test):
    name = 'Air_Classification_BTF'
    features = ['Common Neighbor', 'Salton', 'Jaccard', 'Sorensen', 'Hub Promoted',
               'Hub Depressed', 'Leicht Holme Newman', 'Preferential Attachment',
               'Adamic Adar', 'Resource Allocation', 'Local Path']
    simultaneous_test(train, test, features, best_params, name = name)
    nonsimultaneous_test(train, test, features, best_params, name = name)
    all_shap_values(train, test, features, best_params, name = name)

def WTF(train, test):
    name = 'Air_Classification_WTF'
    features = []
    for c in data.columns:
        if  'Weighted' in c:
            features.append(c)
    simultaneous_test(train, test, features, best_params,name = name)
    nonsimultaneous_test(train, test, features, best_params, name = name)
    all_shap_values(train, test, features, best_params, name = name)

def WWW(train, test):
    name = 'Air_Classification_WWW'
    features = ['Curr FWeight']
    simultaneous_test(train, test, features, best_params, name = name)
    nonsimultaneous_test(train, test, features, best_params, name = name)
    all_shap_values(train, test, features, best_params, name = name)

def BTFW(train, test):
    name = 'Air_Classification_BTFW'
    features = ['Common Neighbor', 'Salton', 'Jaccard', 'Sorensen', 'Hub Promoted',
               'Hub Depressed', 'Leicht Holme Newman', 'Preferential Attachment',
               'Adamic Adar', 'Resource Allocation', 'Local Path','Curr FWeight']

    simultaneous_test(train, test, features, best_params, name = name)
    nonsimultaneous_test(train, test, features, best_params, name = name)
    all_shap_values(train, test, features, best_params, name = name)

if __name__ == "__main__":
    # to run classification on bus change the data
    # data =

    global best_params
    best_params = None
    for data_path in ['../data/features/bus_2005_2015.csv','../data/features/usair_2004_2021.csv']:
        data = pd.read_csv(data_path)
        train, test = get_edge_slice(data)
        BTF(train, test)
        WTF(train, test)
        WWW(train, test)
        BTFW(train, test)
