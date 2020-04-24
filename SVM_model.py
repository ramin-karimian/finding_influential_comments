from sklearn import svm
import pandas as pd
import pickle
from sklearn.metrics import precision_recall_fscore_support , balanced_accuracy_score , classification_report
# from utils import *
from imblearn.over_sampling import RandomOverSampler ,  SVMSMOTE
from imblearn.pipeline import make_pipeline
from utils_funcs.utils import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def model(Train_X_tfidf , Test_X_tfidf , Train_Y, Test_Y ):
    print("Training started")
    SVM=svm.LinearSVC(C=1.0,multi_class='ovr')
    # SVM=svm.LinearSVC()
    # SVM=svm.SVC(decision_function_shape='ovo')
    SVM.fit(Train_X_tfidf,list(Train_Y))
    print("Testing started")
    pred=SVM.predict(Test_X_tfidf)
    b_a = balanced_accuracy_score(list(Test_Y), pred)
    c_a = classification_report(list(Test_Y), pred)
    weighted_percision,weighted_recal,weighted_f1,_ = precision_recall_fscore_support(list(Test_Y), pred, average='macro')
    print(f"Test:\n weighted_percision: {weighted_percision}\n weighted_recal: {weighted_recal}\n weighted_f1: {weighted_f1}")
    print(b_a)
    print(c_a)
    return pred

# def prepare_data(datapath):
#     df = load_data(datapath)[0]
#     # X_train, X_test, y_train, y_test = train_test_split(
#     #     list([' '.join(x) for x in df["tokens"]]),
#     #     to_categorical(df["Class"]-1,num_classes=len(df["Class"].unique())),
#     #     test_size=0.3, shuffle=True)
#     X_train, X_test, y_train, y_test = train_test_split(
#         # list([' '.join(x) for x in df["tokens"]]),
#         list([' '.join(x) for x in df["tokens"]]),
#         df["Class"],
#         test_size=0.8, shuffle=True)
#
#     return X_train, X_test, y_train, y_test

def prepare_data(datapath):
    X_train, X_test, y_train, y_test = load_data(datapath)[0]
    # X_train, X_test, y_train, y_test = train_test_split(
    #     list([' '.join(x) for x in df["tokens"]]),
    #     to_categorical(df["Class"]-1,num_classes=len(df["Class"].unique())),
    #     test_size=0.3, shuffle=True)
    return X_train, X_test, y_train, y_test

# def tokenize(config,training_sentences,testing_sentences):
#     tokenizer = Tokenizer(num_words=config["vocab_size"],oov_token = config["oov_tok"])
#     tokenizer.fit_on_texts(training_sentences)
#     word_index = tokenizer.word_index
#     sequences = tokenizer.texts_to_sequences(training_sentences)
#     padded = pad_sequences(sequences,maxlen=config["max_length"],
#                               truncating=config["trunc_type"])
#
#     testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
#     testing_padded = pad_sequences(testing_sequences,maxlen=config["max_length"],
#                                    truncating=config["trunc_type"])
#     return word_index, padded , testing_padded

if __name__=="__main__":
    config={
        "vocab_size" : 1000000,
        "embedding_dim" : 16,
        # "embedding_dim" : 300,
        "max_length" : 80,
        'trunc_type' :'post',
        "oov_tok" : "<OOV>",
        "num_epochs": 5,
        "emb_path" : "data/source_data/embeddings_index(from_GoogleNews-vectors-negative300).pkl"
    }
    # with open("data/tfidf_vectorized_data.pkl","rb") as f:
    #     Train_X_tfidf , Test_X_tfidf , Train_Y, Test_Y = pickle.load(f)
    # datapath = f"data/preprocessed_tagged_data.pkl"
    datapath = f"data/tfidf_vectorized_preprocessed_tagged_data.pkl"
    training_sentences , testing_sentences , Train_Y, Test_Y = prepare_data(datapath)

    # word_index, padded , testing_padded = tokenize(config,training_sentences,testing_sentences)
    # pred= model(padded, testing_padded, Train_Y, Test_Y )

    pred= model(training_sentences, testing_sentences, Train_Y, Test_Y )

    # pred= model(padded, padded, Train_Y, Train_Y )
    # pred= model(X_resampled , Test_X_resampled , y_resampled, Test_y_resampled )
