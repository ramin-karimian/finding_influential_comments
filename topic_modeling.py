import os
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from utils_funcs.utils import *

def lda_model(confiq,common_texts,limit):
    # common_texts=common_texts[:100]
    common_dictionary = Dictionary(common_texts)
    # common_dictionary.filter_extremes(no_below=1, no_above=limit)
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    lda = LdaMulticore(corpus=common_corpus, num_topics=confiq["num_topics"],
                       id2word=common_dictionary, workers=confiq["num_cores"],
                       alpha=confiq["alpha"], eta=confiq["eta"],
                       minimum_probability=confiq["minimum_probability"])
    return lda , common_corpus , common_dictionary

def prepare_result(lda, df, common_corpus, num_topics,c):
    df["topic_distribution"]=None
    for i in range(len(common_corpus)):
        if i%2000==0:
            print(i)
        # df["topic_distribution"][i]=lda[common_corpus[i]]
        df["topic_distribution"][c+i]=lda[common_corpus[i]]
        # df.loc[df.index[i],"topic_distribution"]=lda[common_corpus[i]]
        ld=lda[common_corpus[i]]
        for j in range(num_topics):
            if "topic"+str(j) not in df.columns:
                df["topic"+str(j)]=None
            # df["topic"+str(j)][i]=ld[j][1]
            df["topic"+str(j)][c+i]=ld[j][1]
            # df.loc[df.index[i],"topic"+str(j)]=ld[j][1]
    return df

def prepare_excel_output(df,lda,num_topics,num_words):
    df2=pd.DataFrame(df,
                     columns=["topic"+str(i) for i in range(num_topics)],
                     index=range(num_words))

    for t in range(num_topics):
        ser=pd.Series([])
        for x in lda.get_topic_terms(t,num_words):
            ser[lda.id2word[x[0]]]=x[1]
        df2["topic"+str(t)] = list(ser.sort_values()[::-1].keys())

    df1=pd.DataFrame(df,columns=["topic"+str(i) for i in range(num_topics)])
    return df1,df2

if __name__=="__main__":
    datapath="data/preprocessed_data(polarity_added).pkl"
    confiq={"num_topics":6,
        "oneOrTotal" : ["one_article","total","total_one_article"][0],
        "num_cores":5,
        # "alpha":1e-5,
        "alpha":1/6,
        "eta":5e-1,
        "minimum_probability":0.0,
        "num_words":40,
        "limit":0.9}

    modelName=f"lda_model_{confiq['num_topics']}_{confiq['oneOrTotal']}"
    dirname=f"models/{modelName}"
    os.mkdir(dirname)


    # common_texts, df = load_data(datapath,extention='tokens', article=confiq["oneOrTotal"])
    data = load_data(datapath,extention='tokens', article=confiq["oneOrTotal"])
    cc=0
    c=0
    for t in data:
        common_texts, df = t[0], t[1]
        print(df.iloc[0]['articleID'])
        c= c+1
        print(c)
        lda , common_corpus, common_dictionary = lda_model(confiq,common_texts,confiq['limit'])
        df = prepare_result(lda, df, common_corpus, confiq["num_topics"],cc)
        # print(df.iloc[:]["topic_distribution"])
        df1, df2 = prepare_excel_output(df,lda,confiq["num_topics"],confiq["num_words"])
        # print(df["topic_distribution"])
        artId = df.iloc[0]['articleID'].split("/")[-1]
        filename = f"/{modelName}_({artId})"
        newdirname = dirname + f"/{artId}"
        os.mkdir(newdirname)
        savepath = newdirname+filename+".pkl"
        lda.save(newdirname+filename)
        # print(df["topic_distribution"])
        save_data(savepath, df)
        # print(df["topic_distribution"])

        df1.to_excel(f"{newdirname}/topic_distro_{confiq['num_topics']}_{confiq['limit']}_({artId}).xlsx")
        df2.to_excel(f"{newdirname}/topic_words_{confiq['num_topics']}_{confiq['limit']}_({artId}).xlsx")
        cc= cc + len(df)
