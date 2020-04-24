from sklearn.cluster import KMeans
from utils_funcs.utils import *

def model(confiq,df,data,df_total,embpath= None):
    # emb = load_data(embpath)[0]
    # res = KMeans(confiq["n_clusters"],random_state=0).fit(emb)
    res = KMeans(confiq["n_clusters"],random_state=0).fit(data)
    cluster_counts = pd.Series.value_counts(res.labels_)
    df[confiq["n_clusters"]]=cluster_counts
    df_total[f"cluster_({confiq['n_clusters']})"] = res.labels_
    return res, df , df_total

def output_results(writer,n_clusters,df):
    df.to_excel(writer, sheet_name=f"{n_clusters}")


if __name__=="__main__":
    oneOrTotal = ["one_article","total","total_one_article"][0]
    # modelName = f"my_word2vec_model_{oneOrTotal}"

    num_topics = 6
    modelName = f"lda_model_{num_topics}_{oneOrTotal}"
    artId = f"451560ec-e0ff-5977-955c-e52c34e85d80"
    datapath = f"models/{modelName}/{artId}/{modelName}_({artId}).pkl"
    data, df_total = load_data(datapath,extention = "topics",article=oneOrTotal)[0]
    # data, errlist = load_data(datapath,extention = "topics",article=oneOrTotal)[0]

    # datapath=f"models/{modelName}/{modelName}.pkl"
    # data = load_data(datapath)[0]
    # artId= data["articleID"][0].split("/")[-1]
    # embpath=f"models/{modelName}/{modelName}_embeddings_({artId}).pkl"

    cf=[5,7,8,10,12,15]
    df = pd.DataFrame(columns=cf)
    df[cf[0]] = [None]*max(cf)

    # writer = pd.ExcelWriter(f'models/{modelName}/results.xlsx',
    #                         engine='xlsxwriter')
    for c in cf:
        print(c)
        confiq={"n_clusters":c}
        # datapath=f"models/{modelName}/{modelName}.pkl"
        res,df,df_total= model(confiq,df,data,df_total)
        # output_results(writer,c,df)
    df.to_excel(f'models/{modelName}/{artId}/multi_clustring_results.xlsx')
    save_data(f'models/{modelName}/{artId}/{modelName}.pkl',df_total)
    df_total.to_csv(f'models/{modelName}/{artId}/{modelName}.csv')
    # writer.save()


# if __name__=="__main__":
#     tl=[20,30,40,50,60,70,100,300,400,450,600,1000]
#     # tl=[20,30,40,50,60,70,100,300,600,1000]
#     cf=[10,15,20,25,30,35,40]
#     # tl=[20,30]
#     # cf=[20,25]
#     df = pd.DataFrame(columns=tl)
#     writer = pd.ExcelWriter(f'results.xlsx', engine='xlsxwriter')
#     for c in cf:
#         print(c)
#         df = pd.DataFrame(columns=tl)
#         for num_topics in tl:
#             confiq={"n_clusters":c,
#                     "num_topics":num_topics}
#             modelName=f"lda_model_{num_topics}"
#             # datapath=f"models/{modelName}/{modelName}.pkl"
#             datapath="models/my_word2vec_model_one_article/embeddings"
#             res,df= model(confiq,datapath,df)
#         output_results(writer,c,df)
#     writer.save()


