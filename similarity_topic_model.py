from backup.scripts.utils import *
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
    confiq={"num_topics":50}
    oneOrTotal=["one_article","total"][1]
    modelName = f"lda_model_{confiq['num_topics']}_{oneOrTotal}"
    dirname=f"models/{modelName}"
    filename=f"/{modelName}"
    datapath= dirname+filename+".pkl"
    fname=dirname+filename+"_topical_similarities.pkl"

    # data, df = load_data(datapath,extention="topics",article="one_article")
    data, df = load_data(datapath,extention="topics",article=oneOrTotal)
    data = [[i,np.array(x).reshape(1,-1)] for i,x in enumerate(data)]
    # data = data[:5]
    sims = similarity(data)
    sims=pd.DataFrame(sims,index = df["commentID"], columns= df["commentID"])
    with open(fname,"wb") as f:
        pickle.dump(sims,f)
        sims.to_csv(fname[:-4]+".csv")
