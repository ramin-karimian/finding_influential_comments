import os
from sklearn.metrics.pairwise import cosine_similarity
from utils_funcs.utils import *
from utils_funcs.ESA_model import *
from time import time as tm

def interpret_data(se,data):
    modif_data=[]
    ti=tm()
    errlist=[]
    for i in range(len(data)):
        check_print(i,step=100,time=ti)
        try :
            modif_data.extend(se.interpretation_vector(" ".join(data[i])).toarray())
        except:
            errlist.append(i)
            print("error in i:", i )

    # data = [ se.interpretation_vector(" ".join(x)).toarray() for x in data]
    # return data
    return modif_data,errlist

def similarity( data):
    sims = cosine_similarity([x for x in data], [x for x in data])
    return sims

if __name__=="__main__":
    datapath="data/preprocessed_data(polarity_added).pkl"
    # emb_path= "data/source_data/embeddings_index(from_GoogleNews-vectors-negative300).pkl"
    oneOrTotal = ["one_article","total","total_one_article"][0]
    dirname = f"ESA_model_{oneOrTotal}"
    path = f"models/"+dirname

    data , df = load_data(datapath,extention = "tokens",article=oneOrTotal)[0]
    data=data[:110]
    df = df.iloc[:110]
    artId= df["articleID"][0].split("/")[-1]
    if dirname not in os.listdir("models"):
        os.mkdir(path)
    path = path + f"/{artId}"
    os.mkdir(path)

    simspath = path + f"/{dirname}_similarities_({artId}).pkl"
    vecspath = path + f"/{dirname}_vectors_({artId}).pkl"
    # dfpath = path + f"/{dirname}.pkl"

    # emb , df =embs(data,emb_path)

    se = SemanticAnalyser()
    print("1 " , np.shape(data))
    data ,errlist= interpret_data(se,data)
    save_data(vecspath,[data,errlist])
    print("2 " ,np.shape(data))
    sims = similarity(data)
    print("3 ",  np.shape(data))
    sims = pd.DataFrame(sims,index = df["commentID"], columns= df["commentID"])
    save_data(simspath,sims)
    # save_data(dfpath,df)
    sims.to_csv(simspath[:-4]+".csv")
    # df.to_csv(dfpath[:-4]+".csv")

