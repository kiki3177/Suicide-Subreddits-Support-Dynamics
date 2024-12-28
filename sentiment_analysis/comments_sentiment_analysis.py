import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

suicide_comments = pd.read_csv("SuicideWatch_comments_cleaned.csv", dtype='str', encoding='utf-8', lineterminator='\n')
depression_comments = pd.read_csv("depression_comments_cleaned.csv", dtype='str', encoding='utf-8', lineterminator='\n')

suicide_comments['sentiment_score'] = [analyzer.polarity_scores(str(x)) for x in suicide_comments['content']]
depression_comments['sentiment_score'] = [analyzer.polarity_scores(str(x)) for x in depression_comments['content']]

suicide_comments.to_csv('/content/drive/MyDrive/research_project/SuicideWatch_comments_cleaned.csv', index=False)
depression_comments.to_csv('/content/drive/MyDrive/research_project/depression_comments_cleaned.csv', index=False)
