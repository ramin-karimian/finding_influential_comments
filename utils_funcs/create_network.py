import networkx as nx
from time import time as tm
# from utils_funcs.multi_processing import multi_process , one_processing_func

def create_network(data,threshold):
    edgeList=[]
    c = 0
    # for i in range(len(data)):
    cols = list(data.columns)
    indxs = list(data.index)
    for i in cols:
        c= c +1
        # check_print(c)
        for j in indxs[cols.index(i)+1:]:
            th = data[i][j]
            if th >= threshold:
                edgeList.append((i,j))
                # g.add_edge(i,j,th=th)
    # print(f"network creation took {tm()-t1} s")
    # return g
    return edgeList




# def create_network(data,threshold):
#     t1=tm()
#     g = nx.Graph()
#     c = 0
#     # for i in range(len(data)):
#     cols = list(data.columns)
#     indxs = list(data.index)
#     for i in cols:
#         c= c +1
#         check_print(c)
#         for j in indxs[cols.index(i)+1:]:
#             th = data[i][j]
#             if th >= threshold:
#                 g.add_edge(i,j,th=th)
#     print(f"network creation took {tm()-t1} s")
#     return g