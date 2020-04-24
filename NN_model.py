import tensorflow as  tf
import os
# tf.enable_eager_execution()
# import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils_funcs.utils import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support , balanced_accuracy_score , classification_report
from imblearn.over_sampling import RandomOverSampler
# def prepare_data():
#     # imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
#     # train_data, test_data = imdb['train'], imdb['test']
#     train_data, test_data  = imdb.load_data()
#
#     training_sentences = []
#     training_labels = []
#
#     testing_sentences = []
#     testing_labels = []
#     # str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
#     for s,l in zip(train_data[0],train_data[1]):
#         training_sentences.append(str(s.tonumpy()))
#         training_labels.append(l.tonumpy())
#
#     for s,l in zip(test_data[0],test_data[1]):
#         testing_sentences.append(str(s.tonumpy()))
#         testing_labels.append(l.tonumpy())
#
#     training_labels_final = np.array(training_labels)
#     testing_labels_final = np.array(testing_labels)
#
#     return training_sentences, training_labels_final, testing_sentences, testing_labels_final


# def prepare_data(df):
#
#     # df = shuffle(df)
#     print(df["Class"][0])
#     labels = to_categorical(df["Class"]-1,num_classes=len(df["Class"].unique()))
#     print(df["Class"][0])
#     # total_X_train, total_X_test, total_y_train, total_y_test = [], [], [], []
#
#
#     total_X_train, total_X_test, total_y_train, total_y_test = [], [], [], []
#     X_train, X_test, y_train, y_test = train_test_split(
#         list([' '.join(x) for x in df["tokens"][:150]]),
#         labels[:150],
#         test_size=0.3, shuffle=True)
#     for x,y in zip([total_X_train, total_X_test, total_y_train, total_y_test],
#                    [X_train, X_test, y_train, y_test]):
#         x.extend(y)
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         list([' '.join(x) for x in df["tokens"][330:518]]),
#         labels[330:518],
#         test_size=0.2, shuffle=True)
#     for x,y in zip([total_X_train, total_X_test, total_y_train, total_y_test],
#                    [X_train, X_test, y_train, y_test]):
#         x.extend(y)
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         list([' '.join(x) for x in df["tokens"][518:596]]),
#         labels[518:596],
#         test_size=0.1, shuffle=True)
#     for x,y in zip([total_X_train, total_X_test, total_y_train, total_y_test],
#                    [X_train, X_test, y_train, y_test]):
#         x.extend(y)
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         list([' '.join(x) for x in df["tokens"][596:700]]),
#         labels[596:700],
#         test_size=0.3, shuffle=True)
#     for x,y in zip([total_X_train, total_X_test, total_y_train, total_y_test],
#                    [X_train, X_test, y_train, y_test]):
#         x.extend(y)

    # train_data, test_data  = imdb.load_data()
    # training_sentences , training_labels = train_data
    # testing_sentences , testing_labels = train_data
    # return X_train, X_test, np.array(y_train), np.array(y_test)
    # return X_train, X_test, y_train, y_test
    # return total_X_train, total_X_test, np.array(total_y_train), np.array(total_y_test)

def prepare_data(df):

    # df = shuffle(df)
    print(df["Class"][0])
    labels = to_categorical(df["Class"]-1,num_classes=len(df["Class"].unique()))
    print(df["Class"][0])

    X_train, X_test, y_train, y_test = train_test_split(
        list([' '.join(x) for x in df["tokens"]]),
        labels,
        test_size=0.4, shuffle=True)

    return X_train, X_test, y_train, y_test



def tokenize(config,training_sentences,testing_sentences):
    tokenizer = Tokenizer(num_words=config["vocab_size"],oov_token = config["oov_tok"])
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences,maxlen=config["max_length"],
                              truncating=config["trunc_type"])

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences,maxlen=config["max_length"],
                                   truncating=config["trunc_type"])

    return word_index, padded , testing_padded

def decode_review(text,reversed_word_index):
    return " ".join([reversed_word_index.get(i,"?") for i in text ])

def train_model(config,padded,training_labels_final,testing_padded,testing_labels_final,emb):
    model = tf.keras.models.Sequential([
        # keras.layers.Embedding(config["vocab_size"], config["embedding_dim"],
        #                        input_length=config["max_length"] ),
        keras.layers.Embedding(config["vocab_size"],config["embedding_dim"],
                               # weights=load_emb_matrix(config["emb_path"]),
                               weights= [emb],
                               input_length=config["max_length"] ),

        keras.layers.LSTM(128,activation='relu',return_sequences=True,dropout=0.1),
        keras.layers.LSTM(64,activation='relu',return_sequences=False,dropout=0.1),
        keras.layers.Dense(16,activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(4,activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    print(model.summary())
    model.fit(padded,training_labels_final, epochs=config["num_epochs"] ,
              validation_data=(testing_padded,testing_labels_final)
              )
    return model

def visualize(config,weights):
    import io

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for word_num in range(1, config['vocab_size']):
      # word = reverse_word_index[word_num]
      embeddings = weights[word_num]
      # out_m.write(word + "\n")
      out_m.write(str(word_num) + "\n")
      out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()

    try:
      from google.colab import files
    except ImportError:
      pass
    else:
      files.download('vecs.tsv')
      files.download('meta.tsv')

def embedding(embeddings_index , word_index,embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def custom_metrics(Test_Y,pred):
    weighted_percision,weighted_recal,weighted_f1, s  = precision_recall_fscore_support(list(Test_Y), pred, average='macro')
    return weighted_percision,weighted_recal,weighted_f1 , s

if __name__=="__main__":
    config={
        "vocab_size" : 10000,
        # "embedding_dim" : 16,
        "embedding_dim" : 300,
        "max_length" : 120,
        'trunc_type' :'post',
        "oov_tok" : "<OOV>",
        "num_epochs": 200,
        "emb_path" : "data/source_data/embeddings_index(from_GoogleNews-vectors-negative300).pkl"
    }
    datapath = f"data/preprocessed_tagged_data.pkl"
    df = load_data(datapath)[0]
    training_sentences, testing_sentences, training_labels_final, testing_labels_final = prepare_data(df)
    word_index, padded , testing_padded = tokenize(config,training_sentences,testing_sentences)

    ros = RandomOverSampler(random_state=0)
    padded, training_labels_final = ros.fit_resample(padded, training_labels_final)

    config["vocab_size"] = len(word_index) +1
    emb = load_emb_matrix(config["emb_path"])
    emb_index = embedding(emb , word_index, config["embedding_dim"])
    del emb
    model = train_model(config,padded,training_labels_final,testing_padded,testing_labels_final,emb_index)
    pred = model.predict(testing_padded)

    for i in range(len(pred)):
        l=np.zeros(len(pred[0]))
        l[np.argmax(pred[i])]=1
        pred[i]=l
    res = custom_metrics(testing_labels_final,pred)
    # e = model.layers[0]
    # weights = e.get_weights()[0]
    # print(np.shape(weights))
    # visualize(config,weights)
