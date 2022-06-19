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
from features_extractor import features_extractor

def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def text_to_id(text):
    """
    Convert input text to id.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    text = strip_accents(text.lower())
    text = re.sub(r"\d", "", text)
    text=re.sub(r"^\s+", "", text)
    text=re.sub(r"\s+$", "", text)
    text = re.sub(r"\s+","_", text, flags = re.I)
    #text = re.sub('[ ]+', '_', text)
    text = re.sub('[^a-zA-Z_-]', '', text)
    return text

if __name__ == "__main__":
    from_file = '../data/raw_usair_data/*.csv'
    feature_file = '../data/features/usair_2004_2021.csv'

    try:
        data = pd.read_csv(to_file, sep=';')
        data.set_index(['YEAR', 'MONTH'], inplace=True)
    except:
        print(f'{to_file} not found! Generating graphs from raw')
        dfs=[]
        for f in tqdm(sorted(glob(from_file))):
            df=pd.read_csv(f,engine="python",error_bad_lines=False)
            df=df[['YEAR','MONTH','ORIGIN_CITY_NAME','DEST_CITY_NAME','PASSENGERS','DEPARTURES_PERFORMED']]
            df=df.rename(index=str, columns={"ORIGIN_CITY_NAME": "source",
                                             "DEST_CITY_NAME": "target",
                                             'PASSENGERS':'passengers',
                                             'DEPARTURES_PERFORMED':'weight'})
            df['source']=df.apply(lambda row: text_to_id(str(row.source)), axis=1)
            df['target']=df.apply(lambda row: text_to_id(str(row.target)), axis=1)
            df=df.groupby(['YEAR','MONTH','source','target']).sum()
            df=df.reset_index()
            dfs.append(df[df.weight !=0 ])
        data=pd.concat(dfs, ignore_index=True)
        data=data.reset_index().drop(columns='index')
        data.set_index(['YEAR', 'MONTH'],inplace=True)
        data.sort_index(inplace=True)
        data.to_csv(to_file,sep=';')

    data = data[data.source != data.target]
    data = data[data.weight!=0]
    year = list(data.index.get_level_values(0).unique())
    month = list(data.index.get_level_values(1).unique())
    graphs_air = []
    date_air = []
    for y in year:
        for m in month:
            if y==2021 and m==9:
                break
            df = data.loc[y,m]
            date_air.append(date(y,m,1))
            G = nx.from_pandas_edgelist(df, edge_attr=True)
            graphs_air.append(G)
    features = features_extractor(graphs_air, date_air)
    features.to_csv(feature_file)
