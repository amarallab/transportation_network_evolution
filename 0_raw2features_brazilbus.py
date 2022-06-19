import re
import sys
import unicodedata

import pandas as pd
import networkx as nx
import numpy as np

from tqdm import tqdm
from datetime import date
from glob import glob
from collections import defaultdict

sys.path.append('src')
from features_extractor import features_extractor_single

def match_datasets(data):
    allcities=pd.read_csv("/data/buses_list_of_cities.csv",index_col=0,names=["CityUF"],encoding="utf-8")
    allcities["CityUF"]=allcities["CityUF"].str.upper()
    allcities["CityUF"]=allcities["CityUF"].str.strip()
    allcities["CityUF"]=allcities["CityUF"].str.replace(" , ",", ")
    allcities["CityUF"]=allcities["CityUF"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    setallcities=set(list(allcities.CityUF.astype(str)))
    data=data[data['ORIGEM'].isin(list(setallcities))]
    data=data[data['DESTINO'].isin(list(setallcities))]
    return data

def bus_network(year=2010,month=12):
    data=pd.read_csv("/data/clean_brazilbus_data/{}.csv".format(year),index_col=None)
    if month is not False:
        data=data[data.MES==month]
    data=data[data.NUMEROLUGAROFERTADOIDA>0]
    data=match_datasets(data)
    data=data[["ORIGEM","DESTINO",'NUMEROVIAGEMIDA']]
    data = data.groupby(["ORIGEM","DESTINO"]).sum().reset_index()
    data=data[data.NUMEROVIAGEMIDA>0]
    data=data.rename(columns={'ORIGEM':'source','DESTINO':'target','NUMEROVIAGEMIDA':'weight'})
    return data

if __name__ = "__main__":
    feature_file = '/data/features/bus_2005_2015.csv'
    graphs_bus = []
    date_bus = []
    for y in range(2005, 2015):
        for m in range(1, 13):
            try:
                df = bus_network(y, m)
            except:
                break
            date_bus.append(date(y, m, 1))
            G = nx.from_pandas_edgelist(df, edge_attr=True)
            graphs_bus.append(G)
    features = features_extractor(graphs_bus, date_bus)
    features.to_csv(feature_file)
