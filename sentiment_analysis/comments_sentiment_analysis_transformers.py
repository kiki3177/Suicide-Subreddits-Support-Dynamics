import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis", device=0)


def cal_sentiment_score(text, max_length=512):
    # Truncate text to max_length if it exceeds
    truncated_text = text[:max_length]
    result = sentiment_model(truncated_text)
    score = result[0]['score']
    return score if result[0]['label'] == 'POSITIVE' else -score


def calculate_improvement(df):
    tqdm.pandas()
    df['submission_score'] = df['content_submission'].progress_apply(cal_sentiment_score)
    df['comment_score'] = df['content_comment'].progress_apply(cal_sentiment_score)

    df['improvement'] = df['comment_score'] - df['submission_score']
    return df

df = pd.read_csv("../pagerank/depression_submitters_replied_2023.csv")
df = calculate_improvement(df)
df.to_csv("depression_submitters_replied_2023_sentiment_score_transformer.csv", index=False)