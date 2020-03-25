import pandas as pd
import pickle
import numpy as np
from time import time as tm

def load_data(datapath,extention=False,article="total"):
    with open(datapath, "rb") as f:
        df = pickle.load(f)
    if article == "total":
        if extention == "tokens":
            data = list(df["tokens"])
            return [(data, df)]
        elif extention == "topics":
            data = list(df["topic_distribution"])
            data = [[y[1] for y in x] for x in data]
            return [(data, df)]
        else:
            return [df]
    elif article=="one_article":
        ID = df["articleID"][0]
        df = df[df["articleID"]==ID]
        if extention == "tokens":
            data = df["tokens"]
            return [(data, df)]
        elif extention == "topics":
            data = list(df["topic_distribution"])
            data = [[y[1] for y in x] for x in data]
            return [(data, df)]
        else:
            return [df]
    elif article=="total_one_article":
        ids = list(df['articleID'].unique())
        dfs = []
        for i in ids:
            # d = df[df["articleID"]==i]
            d = pd.DataFrame(df.loc[df["articleID"]==i])
            if extention == "tokens":
                data = d["tokens"]
                dfs.append((data, d))
                # return data, df
            elif extention == "topics":
                data = list(d["topic_distribution"])
                data = [[y[1] for y in x] for x in data]
                # return data, df
                dfs.append((data, d))
            else:
                dfs.append(d)
                # return df
        return dfs

def check_print(i,step=1000,time=False):
    if i%step==0:
        if time:
            print(i," time(min): ",(tm()-time)/60)
        else:
            print(i)

def save_data(datapath, data):
    # print(data["topic_distribution"])
    with open(datapath, "wb") as f:
        pickle.dump(data, f)

def embs(df, emb_path):
    data = list(df["tokens"])
    emb_matrix = load_emb_matrix(emb_path)
    w2i = word2idx(data)
    emb = []
    emb_dim = len(emb_matrix["music"])
    emb_keys = emb_matrix.keys()
    w2i_keys = w2i.keys()
    for x in data:
        if len(x)==0:
            print(x)
            continue
        ll = []
        for y in x:
            if y not in w2i_keys:
                ll.append(np.array([x for x in range(emb_dim)]))
            else:
                if y not in emb_keys:
                    ll.append(np.array(emb_matrix["unk"]))
                else:
                    ll.append(np.array(emb_matrix[y]))
        emb.insert(-1,np.array(np.mean(ll, axis=0)))
    print(np.shape(emb))
    df["emb"]=[e for e in emb]
    return emb , df

def load_emb_matrix(emb_path):
    with open(emb_path, "rb") as f:
        emb_matrix = pickle.load(f)
    return emb_matrix

def word2idx(data):
    w2i = {}
    for x in data:
        for y in x:
            if y not in w2i.keys():
                w2i[y] = len(w2i) + 1
    return w2i

