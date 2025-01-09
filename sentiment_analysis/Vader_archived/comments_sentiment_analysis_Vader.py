import pandas as pd
import numpy as np
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def cal_sentiment_score(df):
    tqdm.pandas()
    df["sentiment_score_0"] = df["content_submission"].progress_apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    df["sentiment_score_1"] = df["content_comment"].progress_apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    return df


df = pd.read_csv("../pagerank/suicide_submitters_replied_2023.csv")
cal_sentiment_score(df)

df.to_csv("suicide_submitters_replied_2023_sentiment_score_Vader.csv", index=False)