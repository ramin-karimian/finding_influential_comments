import numpy as np
import pandas as pd
import pickle
import os
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaMulticore
from utils import *


# Create a corpus from a list of texts

# common_texts = [
#                 ['computer', 'time', 'graph'],
#                 ['survey', 'response', 'eps'],
#                 ['human', 'system', 'computer']
#               ]
datapath="pickles/preprocessed_data.pkl"
common_texts, df = load_data(datapath)
# common_texts=common_texts[:1000]
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

# Train the model on the corpus.
confiq={"num_topics":500,
        "num_cores":10,
        "alpha":1e-5,
        "eta":5e-1,
        "minimum_probability":0.0}
modelName=f"lda_model_{confiq['num_topics']}"
# lda = LdaModel(common_corpus, num_topics=10)
if __name__=="__main__":
    lda = LdaMulticore(corpus=common_corpus, num_topics=confiq["num_topics"], id2word=common_dictionary, workers=confiq["num_cores"], alpha=confiq["alpha"], eta=confiq["eta"],
                  minimum_probability=confiq["minimum_probability"])
    dirname=f"models/{modelName}"
    os.mkdir(dirname)
    filename=f"/{modelName}"
    lda.save(dirname+filename)
    df["topic_distribution"]=None
    print("1")
    for i in range(len(common_corpus)):
        if i%2000==0:
            print(i)
        df["topic_distribution"][i]=lda[common_corpus[i]]
        ld=lda[common_corpus[i]]
        for j in range(confiq["num_topics"]):
            if "topic"+str(j) not in df.columns:
                df["topic"+str(j)]=None
            df["topic"+str(j)][i]=ld[j][1]
    with open(dirname+filename+".pkl","wb") as f:
        pickle.dump(df,f)
        df1=pd.DataFrame(df,columns=["topic"+str(i) for i in range(confiq["num_topics"])])
        df1.to_excel(f"models/{modelName}/test{confiq['num_topics']}.xlsx")

