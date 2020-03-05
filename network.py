import networkx as nx
import itertools
from time import time as tm
from utils import *
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import  greedy_modularity_communities , label_propagation_communities

def create_network(data,threshold):
    t1=tm()
    g = nx.Graph()
    c = 0
    # for i in range(len(data)):
    for i in data.columns:
        c= c +1
        check_print(c)
        for j in data.index:
            th = data[i][j]
            if th >= threshold:
                g.add_edge(i,j,th=th)
    print(f"network creation took {tm()-t1} s")
    return g

def func(g,f,return_dict):
    t1 =tm()
    res = f(g)
    d={}
    for k,v in list(res.items()):
        d[k]=v
    print(f.__name__ , f" took {tm()-t1} s")
    return_dict[f.__name__]=d

def modify(return_dict):
    # data = load_data("data/preprocessed_data(polarity_added).pkl",article="one_article")
    data = load_data("data/preprocessed_data(polarity_added).pkl")
    data_dict={}
    for n in list(g.nodes()): data_dict[n]={}
    for i in return_dict.keys():
        for k,v in return_dict[i].items():
            data_dict[k][i]=v
            data_dict[k]["commentBody"]= data["commentBody"][data["commentID"]==k].values[0]
            data_dict[k]["commentID"]= data["commentID"][data["commentID"]==k].values[0]
            data_dict[k]["articleID"]= data["articleID"][data["commentID"]==k].values[0]
            data_dict[k]["total_compound_polarity"]= data["total_compound_polarity"][data["commentID"]==k].values[0]
            data_dict[k]["pos_polarity"]= data["pos_polarity"][data["commentID"]==k].values[0]
            data_dict[k]["neg_polarity"]= data["neg_polarity"][data["commentID"]==k].values[0]
            data_dict[k]["neu_polarity"]= data["neu_polarity"][data["commentID"]==k].values[0]
    return data_dict

def community_detection_1(g,return_dict):
    centrality_communities = girvan_newman(g, most_valuable_edge=None)
    return centrality_communities

def community_detection(g,return_dict):
    # centrality_communities = girvan_newman(g, most_valuable_edge=None)
    t1=tm()
    com = greedy_modularity_communities(g)
    com = list(com)
    for i in range(len(com)):
        for j in com[i]:
            return_dict[j]["greedy_modularity_communities"] = i
    # com = label_propagation_communities(g)
    print(greedy_modularity_communities.__name__,f" took {tm()-t1} s")
    return com , return_dict

def func_community(g,f,return_dict):
    res = f(g)
    name = f.__name__
    if name =="greedy_modularity_communities":
        res = list(res)
    elif name == "":
        1
    return_dict[f.__name__]=res

def create_net_file(g,return_dict,savepath):
    for i,k in return_dict.items():
        for ii,kk in k.items():
            g.nodes[i][ii]=kk
    nx.write_gexf(g,savepath[:-4]+".gexf")

if __name__ == "__main__":
    mode=["create_network","community_detection"][0]
    threshold= 0.5
    # name="doc2vec"
    # datapath = "models/my_doc2vec_model/my_doc2vec_model_vec20_epochs10_cosine_similarities.pkl"
    # name = "word2vec"
    # datapath = "models/my_word2vec_model/word2vec_cosine_similarities.pkl"
    name = "topic_model"
    num_topics=30
    datapath = f"models/lda_model_{num_topics}/lda_model_{num_topics}_topical_similarities.pkl"
    savepath=f"{datapath.split('/')[0]}/{datapath.split('/')[1]}/network_{name}_{threshold}th.pkl"
    print(f"{mode} _ {threshold} _ {name} _\ndatapath {datapath}_\nsavepath {savepath}" )
    if mode=="create_network":
        print(name,"  ",threshold)
        data = load_data(datapath)
        # data=data[:10]
        g = create_network(data,threshold)
        print(len(g.edges()))
        print(f"order:\n betweenness={(len(g.edges())*len(g.nodes())*5*(10**(-7)))/3600} h")
        print(f"order:\n greedy_modularity_communities={((len(g.edges())**3)*5*(10**(-7)))/3600} h")
        funcList=[nx.degree_centrality,nx.clustering,
                  nx.closeness_centrality, nx.betweenness_centrality,
                  nx.pagerank, nx.eigenvector_centrality_numpy,
                   ]
        return_dict = multi_process(g,func,funcList)
        return_dict = modify(return_dict)
        save_data(savepath,[return_dict,g])
        pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")
    elif mode=="community_detection":
        return_dict,g = load_data(savepath)
        print(f"order:\n greedy_modularity_communities={((len(g.edges())**3)*5*(10**(-7)))/3600} h")
        com , return_dict= community_detection(g,return_dict)
        save_data(savepath,[return_dict,g])
        pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")
        create_net_file(g,return_dict,savepath)
        # funcList=[girvan_newman,
        #           greedy_modularity_communities,
        #           label_propagation_communities
        #            ]
        # res_dict = multi_process(g,func_community,funcList)

        # com = community_detection(g,return_dict)


