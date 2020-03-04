import os
from utils import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

datapath="data/preprocessed_data.pkl"
df = load_data(datapath)

if __name__=="__main__":
    sentiment=SentimentIntensityAnalyzer()
    df["total_compound_polarity"]=None
    df["pos_polarity"]=None
    df["neg_polarity"]=None
    df["neu_polarity"]=None
    for i in range(len(df)):
        score=sentiment.polarity_scores(" ".join(df["tokens"][i]))
        df["neg_polarity"][i],df["neu_polarity"][i], df["pos_polarity"][i],df["total_compound_polarity"][i]= score.values()
    # df["polarity"] = df["tokens"].apply(lambda x: sentiment.polarity_scores(" ".join(x)))
    df.to_excel("data/preprocessed_data(polarity_added).xlsx")
    save_data("data/preprocessed_data(polarity_added).pkl",df)

