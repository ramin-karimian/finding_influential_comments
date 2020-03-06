import pandas as pd
import pickle
import numpy as np

def load_data(datapath,extention=False,article="total"):
    with open(datapath, "rb") as f:
        df = pickle.load(f)
    if article == "total":
        if extention == "tokens":
            data = list(df["tokens"])
            return data, df
        elif extention == "topics":
            data = list(df["topic_distribution"])
            data = [[y[1] for y in x] for x in data]
            return data, df
        else:
            return df
    elif article=="one_article":
        ID = df["articleID"][0]
        df = df[df["articleID"]==ID]
        if extention == "tokens":
            data = df["tokens"]
            return data, df
        elif extention == "topics":
            data = list(df["topic_distribution"])
            data = [[y[1] for y in x] for x in data]
            return data, df
        else:
            return df

def check_print(i):
    if i%1000==0:
        print(i)

def save_data(datapath, data):
    with open(datapath, "wb") as f:
        pickle.dump(data, f)

def embs(data, emb_path):
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
    return emb

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

