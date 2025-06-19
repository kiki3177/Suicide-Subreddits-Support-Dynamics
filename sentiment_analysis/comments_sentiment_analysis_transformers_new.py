import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# GoEmotions setup
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
go_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

device = torch.device("mps")
zero_shot = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=device)
zero_shot_labels = ["substantive support", "casual politeness"]

positive_labels = ['caring', 'gratitude', 'joy', 'love', 'optimism', 'approval']
empathy_labels = ['sadness', 'grief', 'fear']

def get_goemotions_probs(text):
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    return probs

def classify_reply(text):
    result = zero_shot(str(text), candidate_labels=zero_shot_labels)
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    return pd.Series([top_label, top_score])

def compute_word_count(text):
    return len(str(text).split())

def analyze_comments(df, alpha=0.5):
    tqdm.pandas()
    df['goemotions_probs'] = df['content_comment'].progress_apply(get_goemotions_probs)
    df['word_count'] = df['content_comment'].apply(compute_word_count)
    df[['reply_type', 'reply_confidence']] = df['content_comment'].progress_apply(classify_reply)
    df['classification_weight'] = df['reply_type'].apply(lambda x: 1.5 if x == 'substantive support' else 1.0)

    def compute_adjusted_scores(row):
        probs = row['goemotions_probs']
        pos_indices = [go_labels.index(lbl) for lbl in positive_labels]
        emp_indices = [go_labels.index(lbl) for lbl in empathy_labels]
        pos_sum = np.sum(probs[pos_indices])
        emp_sum = np.sum(probs[emp_indices])
        length_factor = row['word_count'] ** alpha
        weight = row['classification_weight']
        adjusted_positive = pos_sum * length_factor * weight
        adjusted_empathy = emp_sum * length_factor * weight
        adjusted_total = adjusted_positive + adjusted_empathy
        return pd.Series([pos_sum, emp_sum, adjusted_positive, adjusted_empathy, adjusted_total])

    df[['positive_sum', 'empathy_sum', 'adjusted_positive', 'adjusted_empathy', 'adjusted_total']] = df.apply(compute_adjusted_scores, axis=1)
    return df


subreddit_names = ["suicide", "depression"]
years = [2023]
core_or_random_list = ["random", "core"]

for subreddit_name in subreddit_names:
    for year in years:
        for core_or_random in core_or_random_list:
            df_path = f"target_commenters_new/{subreddit_name}_{core_or_random}_comments_{year}.csv"
            df = pd.read_csv(df_path)
            df = analyze_comments(df)

            output_path = f"transformer_results_new/{subreddit_name}_{core_or_random}_comments_{year}_sentiment_analysis.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved weighted sentiment file to {output_path}")

            df_filter = df[df['reply_confidence'] >= 0.7]
            reply_type_counts = df_filter['reply_type'].value_counts(normalize=True) * 100
            reply_type_percentages = reply_type_counts.to_dict()

            print("Percentage of each reply type (confidence >= 0.7):")
            for label, pct in reply_type_percentages.items():
                print(f"{label}: {pct:.2f}%")

            avg_positive = df_filter['adjusted_positive'].mean()
            avg_empathy = df_filter['adjusted_empathy'].mean()
            avg_total = df_filter['adjusted_total'].mean()
            print(f"Average adjusted positive score (confidence >= 0.7): {avg_positive:.4f}")
            print(f"Average adjusted empathy score (confidence >= 0.7): {avg_empathy:.4f}")
            print(f"Average adjusted total score (confidence >= 0.7): {avg_total:.4f}")
