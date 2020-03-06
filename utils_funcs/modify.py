from backup.scripts.utils import *

def modify(g,return_dict,oneOrTotal):
    data = load_data("data/preprocessed_data(polarity_added).pkl",article=oneOrTotal)
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
