import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import collections
from utils import *
from sklearn.metrics.pairwise import cosine_similarity



def similarity1(data):
    dic={}
    for i in range(len(data)):
        print(i)
        for j in range(i-1,len(data)):
            sim = cosine_similarity(data[i][1],data[j][1])
            if i not in dic.keys():
                dic[i]={}
            if j not in dic.keys():
                dic[j]={}
            dic[i][j]=sim
            dic[j][i]=sim
    return dic

def similarity(data):
    sims = cosine_similarity([x[1][0] for x in data],[x[1][0] for x in data])
    return sims

if __name__=="__main__":
    confiq={"num_topics":600}
    modelName = f"lda_model_{confiq['num_topics']}"
    dirname=f"models/{modelName}"
    filename=f"/{modelName}"
    datapath= dirname+filename+".pkl"
    fname=dirname+filename+"_topical_similarities.pkl"

    data, df = load_data_topics(datapath)
    data = [[i,np.array(x).reshape(1,-1)] for i,x in enumerate(data)]
    # data = data[:5]
    sims = similarity(data)
    with open(fname,"wb") as f:
        pickle.dump(sims,f)
        # pd.DataFrame(sims).to_csv(fname[:-4]+".csv")
