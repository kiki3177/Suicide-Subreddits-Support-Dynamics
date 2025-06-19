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

    # Define thresholds for categorization
    slight_threshold = 0.1
    moderate_threshold = 0.5

    df['sentiment_tracking_category'] = np.select(
        [
            df['improvement'] < -moderate_threshold,
            (df['improvement'] >= -moderate_threshold) & (df['improvement'] < -slight_threshold),
            (df['improvement'] >= -slight_threshold) & (df['improvement'] <= slight_threshold),
            (df['improvement'] > slight_threshold) & (df['improvement'] <= moderate_threshold),
            df['improvement'] > moderate_threshold
        ],
        [
            'significant negative',
            'slight negative',
            'neutral',
            'slight positive',
            'significant positive'
        ],
        default=''
    )

    return df


subreddit_names = ["suicide", "depression"]
years = [2023, 2024]
core_or_random_list = ['core', 'random']

for subreddit_name in subreddit_names:
    for year in years:
        for core_or_random in core_or_random_list:
            df = pd.read_csv(f"target_commenters/{subreddit_name}_{core_or_random}_submitters_replied_{year}.csv")
            df = calculate_improvement(df)
            df.to_csv(f"transformer_results/{subreddit_name}_{core_or_random}_submitters_replied_{year}_sentiment_score_transformer.csv", index=False)