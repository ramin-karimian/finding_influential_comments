import pickle
import pandas as pd
import random as rand
import numpy as np
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from utils_funcs.utils import *
from keras.utils import to_categorical

def vectorize(data):
    data=data.sample(frac=1)
    data["tokens"]=[" ".join(item) for item in data["tokens"]]
    Train_X, Test_X, Train_Y, Test_Y  = \
        model_selection.train_test_split(data["tokens"],
                                         # data["label"],
                                         # to_categorical(df["Class"]-1,num_classes=len(df["Class"].unique())),
                                         df["Class"],
                                         test_size=0.2)
    Tfidf_vec = TfidfVectorizer(max_features=100000000)
    Tfidf_vec.fit(data["tokens"])

    Train_X_tfidf = Tfidf_vec.transform(Train_X)
    Test_X_tfidf = Tfidf_vec.transform(Test_X)

    return Train_X_tfidf , Test_X_tfidf ,  Train_Y, Test_Y , Tfidf_vec

# def load_data(datapath,dict_filename):
#     df = pd.DataFrame(columns=["tokens"])
#     with open(datapath,"rb") as f:
#         data= pickle.load(f)
#     with open("data/"+dict_filename,"rb") as f:
#         # word2idx,idx2word,hashtag_to_index=pickle.load(f)
#         _,_,hashtag_to_index=pickle.load(f)
#     df["tokens"]=[str(item["tokens"]) for item in data]
#     df["label"]=[convert_to_one_hot_vec(item["hashtags"],hashtag_to_index) for item in data]
#     # for x in df["label"]:
#     #     if 1 not in x:
#     #         print("Err")
#     return df


def convert_to_one_hot_vec(Y,hashtag_to_index):
    # y => list of one sentences's hashtags
    # hashtagToIndex => dictionary converting Hashtag to its index in dictionary
#    m=np.shape(Y)[0]

    Y=list(Y)
    c=hashtag_to_index[Y[0]]
#    Y1=np.zeros(len(hashtag_to_index))
#    for i in range(len(Y)):
#     vector=np.zeros(len(hashtag_to_index))
#     for j in range(len(Y)):
#         if Y[j] in hashtag_to_index.keys():
#             vector[hashtag_to_index[Y[j]]]=1
    # return vector
    return c

if __name__=="__main__":


    # datapath=f"data/stanfordnlp_parallel(parents_added)_and_CoreNLPClient(parents_added)_updated_preprocessed_dataset_{package}_(for_mon_4_to_10)_{version}_hastags{num_for_hs}.pkl"
    datapath = f"../data/preprocessed_tagged_data.pkl"
    # dict_filename=f"words-hashtags_dictionary_{package}_hastags{num_for_hs}.pkl"
    # df=load_data(datapath,dict_filename)
    df=load_data(datapath)[0]

    Train_X_tfidf , Test_X_tfidf , Train_Y, Test_Y , Tfidf_vec= vectorize(df)
    with open("../data/tfidf_vectorized_preprocessed_tagged_data.pkl","wb") as f:
        pickle.dump([Train_X_tfidf , Test_X_tfidf , Train_Y, Test_Y],f)
