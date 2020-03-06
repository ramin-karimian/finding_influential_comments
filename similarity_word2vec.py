import os
from backup.scripts.utils import *
from sklearn.metrics.pairwise import cosine_similarity

def similarity(data):
    sims = cosine_similarity([x for x in data],[x for x in data])
    return sims

if __name__=="__main__":
    datapath="data/preprocessed_data(polarity_added).pkl"
    emb_path= "data/source_data/embeddings_index(from_GoogleNews-vectors-negative300).pkl"

    dirname = "my_word2vec_model"
    path = f"models/"+dirname
    if dirname not in os.listdir("models"):
        os.mkdir(path)
    fname = path + "/word2vec_cosine_similarities.pkl"

    data = load_data(datapath,article="one_article")
    emb =embs(list(data["tokens"]),emb_path)
    sims = similarity(emb)
    sims=pd.DataFrame(sims,index = data["commentID"], columns= data["commentID"])
    with open(fname,"wb") as f:
        pickle.dump(sims,f)
        sims.to_csv(fname[:-4]+".csv")
