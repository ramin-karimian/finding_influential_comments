from sklearn.cluster import KMeans
import pickle
from utils import *


def model(confiq,datapath,writer,df):
    data , _=load_data_topics(datapath)
    res = KMeans(confiq["n_clusters"],random_state=0).fit(data)
    cluster_counts = pd.Series.value_counts(res.labels_)
    df[confiq["num_topics"]]=cluster_counts
    return res,df

def output_results(writer,n_clusters,df):
    df.to_excel(writer, sheet_name=f"{n_clusters}")

if __name__=="__main__":
    tl=[20,30,40,50,60,70,100,300,400,450,600,1000]
    # tl=[20,30,40,50,60,70,100,300,600,1000]
    cf=[10,15,20,25,30,35,40]
    # tl=[20,30]
    # cf=[20,25]
    df = pd.DataFrame(columns=tl)
    writer = pd.ExcelWriter(f'results.xlsx', engine='xlsxwriter')
    for c in cf:
        print(c)
        df = pd.DataFrame(columns=tl)
        for num_topics in tl:
            confiq={"n_clusters":c,
                    "num_topics":num_topics}
            modelName=f"lda_model_{num_topics}"
            datapath=f"models/{modelName}/{modelName}.pkl"
            res,df= model(confiq,datapath,writer,df)
        output_results(writer,c,df)
    writer.save()


