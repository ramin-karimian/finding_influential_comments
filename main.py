import networkx as nx
from time import time as tm
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
    # name="doc2vec"
    # datapath = "models/my_doc2vec_model/my_doc2vec_model_vec20_epochs10_cosine_similarities.pkl"
    # name = "word2vec"
    # datapath = "models/my_word2vec_model/word2vec_cosine_similarities.pkl"
    name = "topic_model"
    num_topics=50
    oneOrTotal=["one_article","total"][1]
    datapath = f"models/lda_model_{num_topics}_{oneOrTotal}/lda_model_{num_topics}_{oneOrTotal}_topical_similarities.pkl"
    savepath=f"{datapath.split('/')[0]}/{datapath.split('/')[1]}/network_{name}_{threshold}th.pkl"
    num_processors = 5
    print(f"{mode} _ {threshold} _ {name} _\ndatapath {datapath}_\nsavepath {savepath}" )

    if mode=="create_network":

        print(name,"  ",threshold)
        data = load_data(datapath)
        t=int(np.floor(len(data)/num_processors))
        funcList=[]
        for n in range(num_processors):
            prccName=f"process number = {n+1}"
            print(prccName)
            if n== num_processors-1:
                numbers=range(n*t,len(data))
            else:
                numbers=range(n*t,(n+1)*t)
            funcList.append([
                create_network,
                data.iloc[numbers[0]:numbers[-1]][numbers[0]:]
                            ])
        return_list=network_multi_process(funcList,argList=[threshold])
        g = nx.from_edgelist(return_list)
        save_data(savepath,[None,g])
        print(len(g.edges()))

    elif mode=="centrality":

        _,g = load_data(savepath)
        print(f"order:\n betweenness={(len(g.edges())*len(g.nodes())*(10**(-7)))/3600} h")
        print(f"order:\n greedy_modularity_communities={((len(g.nodes())**3)*(10**(-7)))/3600} h")
        print(f"order:\n Louvain_modularity={((len(g.nodes())**2)*(10**(-7)))/3600} h")
        # funcList=[nx.degree_centrality,nx.clustering, nx.closeness_centrality,
        #           nx.betweenness_centrality,nx.pagerank, nx.eigenvector_centrality_numpy]
        funcList=[nx.degree_centrality, nx.pagerank,
                  nx.eigenvector_centrality_numpy, my_closeness_centrality]

        return_dict = centrality_multi_process(funcList,[g])
        return_dict = one_processing_func(g,betweenness_centrality_parallel,return_dict)
        return_dict = modify(g,return_dict,oneOrTotal)

        save_data(savepath,[return_dict,g])
        pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")

    elif mode=="community_detection":

        return_dict,g = load_data(savepath)
        print(f"order:\n greedy_modularity_communities={((len(g.nodes())**3)*(10**(-7)))/3600} h")
        print(f"order:\n Louvain_modularity={((len(g.nodes())**2)*(10**(-7)))/3600} h")
        return_dict= community_detection(g,return_dict)

        save_data(savepath,[return_dict,g])
        pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")
        create_net_file(g,return_dict,savepath)



