from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import collections
import os
from sklearn.metrics.pairwise import cosine_similarity
from backup.scripts.utils import *



def model(confiq):
    mdl = Doc2Vec(vector_size=confiq["vector_size"],
                  min_count=confiq["min_count"],
                  epochs=confiq["epochs"])
    return mdl
def assess_model(model,train_corpus):
    ranks = []
    second_ranks = []
    first_ranks = []
    print("hi")
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])
        first_ranks.append(sims[0])
    print("bye")
    cn = collections.Counter(ranks)
    ac = cn[0]/(sum(cn.values()))
    print(cn)
    print(ac)
    return cn , ac , first_ranks , second_ranks

def similarity(model,corpus):
    data = [model.infer_vector(x.words) for x in corpus]
    sim = cosine_similarity(data,data)
    return sim

if __name__=="__main__":
    assess=False
    train = True
    # testing=True
    confiq={"vector_size":20,
        "min_count":2,
        "epochs":10}
    datapath="data/preprocessed_data(polarity_added).pkl"
    dirname = "my_doc2vec_model"
    path = f"models\\"+dirname
    if dirname not in os.listdir("models"):
        os.mkdir(path)
    fname = path+f"\\my_doc2vec_model_vec{confiq['vector_size']}_epochs{confiq['epochs']}"
    df = load_data(datapath,article="one_article")
    train_corpus=[TaggedDocument(df["tokens"][i],[df["commentID"][i]]) for i in range(int(0.9*len(df)))]
    test_corpus=[TaggedDocument(df["tokens"][i],[df["commentID"][i]]) for i in range(int(0.9*len(df)),len(df))]
    corpus= [TaggedDocument(df["tokens"][i],[df["commentID"][i]]) for i in range(len(df))]
    if assess:

        model = model(confiq)
        model.build_vocab(train_corpus)
        model.train(train_corpus,total_examples=model.corpus_count,epochs=model.epochs)
        cn_train, ac_train , train_first_ranks , train_second_ranks= assess_model(model,train_corpus)
        cn_test, ac_test , test_first_ranks , test_second_ranks = assess_model(model,test_corpus)
        print(np.mean([x[1] for x in train_first_ranks]))
        print(np.mean([x[1] for x in train_second_ranks]))
        print(np.mean([x[1] for x in test_first_ranks]))
        print(np.mean([x[1] for x in test_second_ranks]))
        # if not testing:
        model.save(fname)
    elif train:
        model = model(confiq)
        model.build_vocab(corpus)
        print("training started")
        model.train(corpus,total_examples=model.corpus_count,epochs=model.epochs)
        print("training finished")
        sims = similarity(model,corpus)
        sims=pd.DataFrame(sims,index = df["commentID"], columns= df["commentID"])
        with open(fname+"_cosine_similarities.pkl","wb") as f:
            pickle.dump(sims,f)
            sims.to_csv(fname[:-4]+"_cosine_similarities.csv")
    else:
        model = Doc2Vec.load(fname)
