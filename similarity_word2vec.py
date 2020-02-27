import numpy as np
import os
from utils import *
from sklearn.metrics.pairwise import cosine_similarity

def similarity(data):
    sims = cosine_similarity([x for x in data],[x for x in data])
    return sims

if __name__=="__main__":
    datapath="pickles/preprocessed_data(polarity_added).pkl"
    emb_path= "pickles/embeddings_index(from_GoogleNews-vectors-negative300).pkl"

    dirname = "my_word2vec_model"
    path = f"models/"+dirname
    if dirname not in os.listdir("models"):
        os.mkdir(path)
    fname = path + "/word2vec_cosine_similarities.pkl"

    data, df = load_data(datapath)
    emb =embs(data,emb_path)
    sims = similarity(emb)

    with open(fname,"wb") as f:
        pickle.dump(sims,f)
        pd.DataFrame(sims).to_csv(fname[:-4]+".csv")
