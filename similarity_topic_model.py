from sklearn.metrics.pairwise import cosine_similarity
from utils_funcs.utils import *
import os


def similarity(data):
    sims = cosine_similarity([x[1][0] for x in data],[x[1][0] for x in data])
    return sims

if __name__=="__main__":
    config ={"num_topics":6,
            "oneOrTotal":["one_article","total","total_one_article"][0],
            }
    # oneOrTotal=["one_article","total","total_one_article"][1]
    modelName = f"lda_model_{config['num_topics']}_{config['oneOrTotal']}"
    # dirname=f"models/{modelName}"
    # filename=f"/{modelName}"
    # data, df = load_data(datapath,extention="topics",article=config['oneOrTotal'])
    dirlist = os.listdir(f"models/{modelName}")
    for artId in dirlist:
        dirpath=f"models/{modelName}/{artId}"
        datapath = f"{dirpath}/{modelName}_({artId}).pkl"
        res = load_data(datapath,extention="topics")
        data, df = res[0][0], res[0][1]
        data = [[i,np.array(x).reshape(1,-1)] for i,x in enumerate(data)]
        sims = similarity(data)
        sims = pd.DataFrame(sims,index = df["commentID"], columns= df["commentID"])
        # artId= df.iloc[0]['articleID'].split("/")[-1]
        # filename=f"/({artId})_{modelName}"
        # datapath= dirname+filename+".pkl"
        # savepath=dirpath+f"/{modelName}_topical_similarities_({artId}).pkl"
        savepath=dirpath+f"/{modelName}_similarities_({artId}).pkl"
        with open(savepath,"wb") as f:
            pickle.dump(sims,f)
            sims.to_csv(savepath[:-4]+".csv")
