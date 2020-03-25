import networkx as nx
from time import time as tm
import os
from utils_funcs.utils import *
from utils_funcs.create_network import create_network
from utils_funcs.modify import modify
from utils_funcs.multi_processing import centrality_multi_process , network_multi_process , one_processing_func
from utils_funcs.create_net_file import create_net_file
from utils_funcs.my_closeness_centerality import my_closeness_centrality
from utils_funcs.my_betweenness_centrality import betweenness_centrality_parallel
from utils_funcs.community_detections import Louvain_modularity_community as community_detection



if __name__ == "__main__":
    print(tm())
    mode=["create_network","centrality","community_detection"][1]
    threshold= 0.5
    oneOrTotal=["one_article","total","total_one_article"][0]
    # name="doc2vec"
    # datapath = "models/my_doc2vec_model/my_doc2vec_model_vec20_epochs10_cosine_similarities.pkl"

    # # name = "word2vec"
    # name = "my_word2vec_model"
    # modelName = f"{name}_{oneOrTotal}"

    # datapath = "models/my_word2vec_model/word2vec_cosine_similarities.pkl"

    # # name = "topic_model"
    # name = "lda_model"
    # num_topics=6
    # modelName = f"{name}_{num_topics}_{oneOrTotal}"

    name = "ESA_model"
    # num_topics=6
    modelName = f"{name}_{oneOrTotal}"



    dirlist = os.listdir(f"models/{modelName}")
    # dirlist = []
    c = 0
    edg_prt=[]
    nod_prt=[]
    for artId in dirlist:
        # datapath = f"models/{modelName}/{artId}/lda_model_{num_topics}_{oneOrTotal}_topical_similarities_({artId}).pkl"
        datapath = f"models/{modelName}/{artId}/{modelName}_similarities_({artId}).pkl"
        savepath=f"models/{modelName}/{artId}/network_{name}_{threshold}th_({artId}).pkl"
        path = f"models/{modelName}/{artId}/{modelName}.pkl"
        # num_processors = 11
        betweenness_processes = 14
        c = c +1
        print("\nc: ",c)
        print(f"{mode} _ {threshold} _ {name} _\ndatapath {datapath}_\nsavepath {savepath}" )

        if mode=="create_network":

            print(name,"  ",threshold)
            data = load_data(datapath)
            g = create_network(data[0],threshold)
            # t=int(np.floor(len(data)/num_processors))
            # funcList=[]
            # for n in range(num_processors):
            #     prccName=f"process number = {n+1}"
            #     print(prccName)
            #     if n== num_processors-1:
            #         numbers=range(n*t,len(data))
            #     else:
            #         numbers=range(n*t,(n+1)*t)
            #     print(numbers)
            #     print(data.iloc[numbers[0]:numbers[-1],numbers[0]:].shape)
            #     funcList.append([
            #         create_network,
            #         # data
            #         data.iloc[numbers[0]:numbers[-1],numbers[0]:]
            #                     ])
            # return_list=network_multi_process(funcList,argList=[threshold])
            # g = nx.from_edgelist(return_list)
            save_data(savepath,[None,g])
            print("edges: ",len(g.edges()))
            print("nodes: ",len(g.nodes()))
            edg_prt.append(len(g.edges()))
            nod_prt.append(len(g.nodes()))

        elif mode=="centrality":

            res = load_data(savepath)
            _,g = res[0]
            print("edges: ",len(g.edges()))
            print("nodes: ",len(g.nodes()))
            edg_prt.append(len(g.edges()))
            nod_prt.append(len(g.nodes()))
            # print(f"order:\n betweenness={(len(g.edges())*len(g.nodes())*(10**(-7)))/3600} h")
            # print(f"order:\n greedy_modularity_communities={((len(g.nodes())**3)*(10**(-7)))/3600} h")
            # print(f"order:\n Louvain_modularity={((len(g.nodes())**2)*(10**(-7)))/3600} h")
            # funcList=[nx.degree_centrality,nx.clustering, nx.closeness_centrality,
            #           nx.betweenness_centrality,nx.pagerank, nx.eigenvector_centrality_numpy]
            funcList=[nx.degree_centrality, nx.pagerank,
                      nx.eigenvector_centrality_numpy, my_closeness_centrality]
            # funcList=[nx.degree_centrality, nx.pagerank, nx.betweenness_centrality,
            #           nx.eigenvector_centrality_numpy, my_closeness_centrality]

            return_dict = centrality_multi_process(funcList,[g])
            return_dict = one_processing_func(g,betweenness_centrality_parallel,return_dict,processes=betweenness_processes)
            return_dict = modify(g,return_dict,path,oneOrTotal)
            # return_dict = modify(g,return_dict,oneOrTotal)

            save_data(savepath,[return_dict,g])
            pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")

        elif mode=="community_detection":

            # return_dict,g = load_data(savepath)
            res= load_data(savepath)
            return_dict,g = res[0]
            print("edges: ",len(g.edges()))
            print("nodes: ",len(g.nodes()))
            edg_prt.append(len(g.edges()))
            nod_prt.append(len(g.nodes()))
            # print(f"order:\n greedy_modularity_communities={((len(g.nodes())**3)*(10**(-7)))/3600} h")
            # print(f"order:\n Louvain_modularity={((len(g.nodes())**2)*(10**(-7)))/3600} h")
            return_dict= community_detection(g,return_dict)

            save_data(savepath,[return_dict,g])
            pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")
            create_net_file(g,return_dict,savepath)

    print(sum(edg_prt))
    print(sum(nod_prt))


