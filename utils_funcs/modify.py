from utils_funcs.utils import *


# def modify(g,return_dict,datapath,oneOrTotal="total"):
#     # data = load_data("data/preprocessed_data(polarity_added).pkl",article=oneOrTotal)
#     data = load_data(datapath,article=oneOrTotal)
#     data = data[0]
#     data_dict={}
#     for n in list(g.nodes()): data_dict[n]={}
#
#     for col in data.columns:
#         for n in g.nodes():
#             data_dict[n][col]=data[col][data["commentID"]==n].values[0]
#
#     for i in return_dict.keys():
#         for k,v in return_dict[i].items():
#             data_dict[k][i]=v
#     return data_dict

def modify(g,return_dict,datapath,oneOrTotal="total"):
    data = load_data("data/preprocessed_data(polarity_added).pkl",article=oneOrTotal)
    # data = load_data(datapath,article=oneOrTotal)
    data = data[0]
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
            # data_dict[k]["cluster_(7)"]= data["cluster_(7)"][data["commentID"]==k].values[0]
    return data_dict
