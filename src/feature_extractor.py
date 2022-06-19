import math

import networkx as nx
import numpy as np
import pandas as pd

from collections import defaultdict
from datetime import date
from tqdm import tqdm
from itertools import product

distant_df = pd.read_csv('../data/us_air_distance.csv')
distant_map = distant_df.set_index(['source_origin','target_origin']).to_dict()['distance']
distant_map.update(distant_df.set_index(['target_origin','source_origin']).to_dict()['distance'])
population_map = pd.read_csv('../data/us_air_population_all.csv').set_index('Unnamed: 0').fillna(0).to_dict()['0']

def get_gravitation(edges):
    def my_divid(a,b):
        if b==0 or a==0:
            return None
        else:
            return a/b
    Gra = []
    for e in edges:
        u, v = e
        d = distant_map.get(e , 0)
        ni = population_map.get(u, 0)
        nj = population_map.get(v, 0)
        Gra.append(my_divid(ni*nj, d**2))
    meanv = np.mean([i for i in Gra if i])
    return [i if i else meanv for i in Gra]

def features_extractor(graphs, dates):
    def local_path(G, nodeList, epsilon = 0.01):
        A = nx.adjacency_matrix(G, nodelist=nodeList, weight = None).todense()
        return (A**2+epsilon*A**3)

    def l3_path(G, nodeList):
        A = nx.adjacency_matrix(G, nodelist=nodeList, weight = None).todense()
        return (A**3)

    def weighted_local_path(G, nodeList, epsilon = 0.01):
        A = nx.adjacency_matrix(G, nodelist=nodeList, weight='weight').todense()
        return (A**2+epsilon*A**3)

    X = defaultdict(list)
    for i in tqdm(range(len(graphs)-1)):
        G, H = graphs[i], graphs[i+1]
        G.add_nodes_from([n for n in H if n not in G])
        H.add_nodes_from([n for n in G if n not in H])
        Hedges = set(H.edges())
        Gedges = list(G.edges())
        nodeList = list(G.nodes())
        nodeIndex = {node: idx for idx,node in enumerate(nodeList)}
        year = dates[i]

        Ki = dict(G.degree())
        Wi = dict(G.degree(weight='weight'))
        LPI = local_path(G, nodeList)
        L3 = l3_path(G, nodeList)
        WLPI = weighted_local_path(G, nodeList)
        Gra = get_gravitation(Gedges)

        added_edges = list(nx.difference(H,G).edges())

        for j, e in enumerate(Gedges):
            u, v = e
            common_ns = list(nx.common_neighbors(G,u,v))
            w_common_ns = sum([min(G[u][z]['weight'], G[v][z]['weight']) for z in common_ns])
            union_ns = set(G.neighbors(u))|set(G.neighbors(v))
            w_union_ns = Wi[u] + Wi[v]- w_common_ns
            if(w_union_ns==0): print(Wi[u] , Wi[v], [min(G[u][z]['weight'], G[v][z]['weight']) for z in common_ns])
            X['Edge'].append(e)
            X['Year'].append(year)

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
            X['L3 Path'].append(L3[nodeIndex[u],nodeIndex[v]])
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

            X['Removed'].append(e not in Hedges)
            X['Gravity'].append(Gra[j])
            X['Curr Weight'].append(G[u][v]['weight'])
            X['Next Weight'].append(H[u][v]['weight'] if e in Hedges else 0)

            X['Curr FWeight'].append(G[u][v]['weight']/G.size(weight='weight'))
            X['Next FWeight'].append(H[u][v]['weight']/H.size(weight='weight') if e in Hedges else 0)

    df = pd.DataFrame(X)
    return(df)
