import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import random
import math
import datetime

from datetime import date
from collections import defaultdict
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import confusion_matrix,balanced_accuracy_score, \
mean_squared_error,r2_score,mean_absolute_error

import sys
sys.path.append("src/")
from plot_style import *

def get_feature_vector(graph):
    def local_path(G, nodeList, epsilon = 0.01):
        A = nx.adjacency_matrix(G, nodelist=nodeList, weight = None).todense()
        return (A**2+epsilon*A**3)

    def weighted_local_path(G, nodeList, epsilon = 0.01):
        A = nx.adjacency_matrix(G, nodelist=nodeList, weight='weight').todense()
        return (A**2+epsilon*A**3)
    X = defaultdict(list)
    G = graph
    Gedges = list(G.edges())
    nodeList = list(G.nodes())
    nodeIndex = {node: idx for idx,node in enumerate(nodeList)}

    Ki = dict(G.degree())
    Wi = dict(G.degree(weight='weight'))
    LPI = local_path(G, nodeList)
    WLPI = weighted_local_path(G, nodeList)
    for j, e in enumerate(Gedges):
        u, v = e
        common_ns = list(nx.common_neighbors(G,u,v))
        w_common_ns = sum([min(G[u][z]['weight'], G[v][z]['weight']) for z in common_ns])
        union_ns = set(G.neighbors(u))|set(G.neighbors(v))
        w_union_ns = Wi[u] + Wi[v]- w_common_ns
        if(w_union_ns==0): print(Wi[u] , Wi[v], [min(G[u][z]['weight'], G[v][z]['weight']) for z in common_ns])
        X['Edge'].append(e)

        X['Common Neighbor'].append(len(common_ns))
        X['Weighted Common Neighbor'].append(w_common_ns)

        X['Salton'].append(len(common_ns)/math.sqrt(Ki[u]*Ki[v]))
        X['Weighted Salton'].append(w_common_ns/math.sqrt(Wi[u]*Wi[v]))

        X['Sorensen'].append(2*len(common_ns)/(Ki[u]+Ki[v]))
        X['Weighted Sorensen'].append(2*w_common_ns/(Wi[u]+Wi[v]))

        X['Hub Promoted'].append(len(common_ns)/min(Ki[u],Ki[v]))
        X['Weighted Hub Promoted'].append(w_common_ns/min(Wi[u],Wi[v]))

        X['Hub Depressed'].append(len(common_ns)/max(Ki[u],Ki[v]))
        X['Weighted Hub Depressed'].append(w_common_ns/max(Wi[u],Wi[v]))

        X['Leicht Holme Newman'].append(len(common_ns)/(Ki[u]*Ki[v]))
        X['Weighted Leicht Holme Newman'].append(w_common_ns/(Wi[u]*Wi[v]))

        X['Preferential Attachment'].append(Ki[u]*Ki[v])
        X['Weighted Preferential Attachment'].append(Wi[u]*Wi[v])

        X['Local Path'].append(LPI[nodeIndex[u],nodeIndex[v]])
        X['Weighted Local Path'].append(WLPI[nodeIndex[u],nodeIndex[v]])
        if len(common_ns)>0:
            X['Resource Allocation'].append(sum([1/Ki[z] for z in common_ns]))
            X['Weighted Resource Allocation'].append(w_common_ns*sum([1/Wi[z] for z in common_ns]))

            X['Adamic Adar'].append(sum([1/math.log(Ki[z]) for z in common_ns]))
            X['Weighted Adamic Adar'].append(w_common_ns*sum([1/math.log(Wi[z]+1) for z in common_ns]))

            X['Jaccard'].append(len(common_ns)/len(union_ns))
            X['Weighted Jaccard'].append(w_common_ns/w_union_ns)
        else:
            X['Resource Allocation'].append(0)
            X['Weighted Resource Allocation'].append(0)
            X['Adamic Adar'].append(0)
            X['Weighted Adamic Adar'].append(0)
            X['Jaccard'].append(0)
            X['Weighted Jaccard'].append(0)

        X['Curr Weight'].append(G[u][v]['weight'])
        X['Curr FWeight'].append(G[u][v]['weight']/G.size(weight='weight'))
    df = pd.DataFrame(X)
    return(df)

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
    return(X, y, df.loc[:, df.columns == 'Next Weight'].to_numpy())

def add_edges(graphs, inp_graph, time_idx):
    GI0,GI1 = graphs[time_idx+1],graphs[time_idx]
    GI0.add_nodes_from([n for n in GI1 if n not in GI0])
    GI1.add_nodes_from([n for n in GI0 if n not in GI1])
    added_edges = list(nx.difference(GI0,GI1).edges())
    return added_edges

def main(year_start):
    if year_start.month==1: return
    data = pd.read_csv('../data/networks/US_Air_2004_2021.csv', sep=';')
    data.set_index(['YEAR', 'MONTH'], inplace=True)
    data = data[data.source != data.target]
    nodes = set(data.source) & set(data.target)
    data = data[data.weight!=0]
    year = list(data.index.get_level_values(0).unique())
    month = list(data.index.get_level_values(1).unique())
    graphs_air = []
    air_dates = []
    for y in year:
        for m in month:
            if y==2021 and m==9:
                break
            df = data.loc[y,m]
            air_dates.append(date(y,m,1))
            G = nx.from_pandas_edgelist(df, edge_attr=True)
            G.add_nodes_from(nodes)
            graphs_air.append(G)

    data = pd.read_csv('../data/features/US_Air_2004_2021.csv')
    out = {}
    idx = air_dates.index(year_start)
    train, test = get_edge_slice(data)
    features = ['Common Neighbor', 'Salton', 'Jaccard', 'Sorensen', 'Hub Promoted',
           'Hub Depressed', 'Leicht Holme Newman', 'Preferential Attachment',
           'Adamic Adar', 'Resource Allocation', 'Local Path']
    X_train, y_train, _ = df_to_XY(train[train.Year==str(year_start)],features)
    ros = RandomUnderSampler()
    X_train,y_train = ros.fit_resample(X_train,y_train)
    model_btf = XGBClassifier()
    model_btf.fit(X_train, y_train)
    model = model_btf
    diff_btf = []
    graphs_btf = [graphs_air[idx]]
    for i in tqdm(range(0,36)):
        G = graphs_btf[i].copy()
        df = get_feature_vector(G)
        edges, X = df['Edge'].to_numpy(),df[features].to_numpy()

        GI0,GI1 = graphs_air[idx+i],graphs_air[idx+i+1]
        GI0.add_nodes_from([n for n in GI1 if n not in GI0])
        GI1.add_nodes_from([n for n in GI0 if n not in GI1])
        real_removal = set(nx.difference(GI0,GI1).edges())

        pred_prob = model.predict_proba(X).T[0]
        added_edges = add_edges(graphs_air, G, idx+i)
        N_add = len(added_edges)
        for u,v in added_edges:
            G.add_edge(u, v, weight=graphs_air[idx+i+1][u][v]['weight'])
        N_remove = G.number_of_edges() - graphs_air[idx+i+1].number_of_edges()

        removal = zip(edges,pred_prob)
        removal = sorted(removal, key = lambda x: x[1])[0:N_remove]
        remove_edges = [i for i,_ in removal]
        diff_btf.append(len(set(remove_edges)&real_removal)/N_remove)
        G.remove_edges_from(remove_edges)
        graphs_btf.append(G.copy())
    best_params = {'lambda': 0.5650701862593042, 'alpha': 0.0016650896783581535,
           'colsample_bytree': 1.0, 'subsample': 0.5, 'learning_rate': 0.009,
           'n_estimators': 625, 'objective':'reg:squarederror','max_depth': 5, 'min_child_weight': 6}
    features = ['Curr Weight']
    X_train, y_train, y_reg = df_to_XY(train[train.Year==str(year_start)],features)
    ros = RandomUnderSampler()
    X_resample, y_resample = ros.fit_resample(X_train,y_train)
    model_www = XGBClassifier()
    model_www.fit(X_resample, y_resample)
    model_reg = XGBRegressor(**best_params)
    model_reg.fit(X_train, y_reg)
    model = model_www
    diff_www = []
    graphs_www = [graphs_air[idx]]
    for i in tqdm(range(0,36)):
        G = graphs_www[i].copy()
        for u,v in G.edges():
            G[u][v]['weight'] = model_reg.predict([G[u][v]['weight']])[0]
        df = get_feature_vector(G)
        edges, X = df['Edge'].to_numpy(),df[features].to_numpy()
        pred_prob = model.predict_proba(X).T[0]
        GI0,GI1 = graphs_air[idx+i],graphs_air[idx+i+1]
        GI0.add_nodes_from([n for n in GI1 if n not in GI0])
        GI1.add_nodes_from([n for n in GI0 if n not in GI1])
        real_removal = set(nx.difference(GI0,GI1).edges())
        added_edges = add_edges(graphs_air, G, idx+i)
        N_add = len(added_edges)
        for u,v in added_edges:
            G.add_edge(u, v, weight=graphs_air[idx+i+1][u][v]['weight'])
        N_remove = G.number_of_edges() - graphs_air[idx+i+1].number_of_edges()
        removal = zip(edges,pred_prob)
        removal = sorted(removal, key = lambda x: x[1])[0:N_remove]
        remove_edges = [i for i,_ in removal]
        diff_www.append(len(set(remove_edges)&real_removal)/N_remove)
        G.remove_edges_from(remove_edges)
        graphs_www.append(G.copy())
    btf_diff = []
    www_diff = []
    for i in range(0,36):
        G = set(graphs_air[idx+i].edges())
        H = set(graphs_btf[i].edges())
        btf_diff.append(len(G & H)/len(G))
        H = set(graphs_www[i].edges())
        www_diff.append(len(G & H)/len(G))
    out[year_start]=(diff_btf,diff_www,btf_diff,www_diff)
    import pickle
    with open(f'../results/'+ f'{str(year_start)}pred_36' +'.pkl', 'wb') as f:
        pickle.dump(out, f)


    out = {}
    idx = air_dates.index(year_start)
    diff_null = []
    graphs_null = [graphs_air[idx]]
    for i in tqdm(range(0,36)):
        G = graphs_null[i].copy()
        edges = list(G.edges())
        GI0,GI1 = graphs_air[idx+i],graphs_air[idx+i+1]
        GI0.add_nodes_from([n for n in GI1 if n not in GI0])
        GI1.add_nodes_from([n for n in GI0 if n not in GI1])
        real_removal = set(nx.difference(GI0,GI1).edges())
        added_edges = add_edges(graphs_air, G, idx+i)
        N_add = len(added_edges)
        for u,v in added_edges:
            G.add_edge(u, v, weight=graphs_air[idx+i+1][u][v]['weight'])
        N_remove = G.number_of_edges() - graphs_air[idx+i+1].number_of_edges()
        remove_edges = random.sample(edges, N_remove)
        diff_null.append(len(set(remove_edges)&real_removal)/N_remove)
        G.remove_edges_from(remove_edges)
        graphs_null.append(G.copy())
    null_diff = []
    for i in range(0,36):
        G = set(graphs_air[idx+i].edges())
        H = set(graphs_null[i].edges())
        btf_diff.append(len(G & H)/len(G))
    out[year_start]=(diff_null,null_diff)
    import pickle
    with open(f'../results/'+ f'{str(year_start)}pred_36_null' +'.pkl', 'wb') as f:
        pickle.dump(out, f)

if __name__ == '__main__':
    data = pd.read_csv('../data/networks/US_Air_2004_2021.csv', sep=';')
    data.set_index(['YEAR', 'MONTH'], inplace=True)
    data = data[data.source != data.target]
    nodes = set(data.source) & set(data.target)
    data = data[data.weight!=0]
    year = list(data.index.get_level_values(0).unique())
    month = list(data.index.get_level_values(1).unique())
    graphs_air = []
    air_dates = []
    for y in year:
        for m in month:
            if y==2021 and m==9:
                break
            df = data.loc[y,m]
            air_dates.append(date(y,m,1))
            G = nx.from_pandas_edgelist(df, edge_attr=True)
            G.add_nodes_from(nodes)
            graphs_air.append(G)

    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(main)(year_start) for year_start in air_dates[:-36])
