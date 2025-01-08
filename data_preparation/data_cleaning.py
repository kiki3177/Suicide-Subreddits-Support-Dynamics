import pandas as pd
import re
import numpy as np
import os


def extract_thread_id(url):
    match = re.search(r'/comments/([a-zA-Z0-9]+)/', url)
    if match:
        if len(match.group(1)) < 3:
            print(url)
        else:
            return match.group(1)
    else:
        print(url)


def process_thread_ids(submissions, comments):
    # Extract thread IDs for submissions and comments
    submissions['thread_id'] = [extract_thread_id(x) for x in submissions['link']]
    comments['thread_id'] = [extract_thread_id(x) for x in comments['link']]

    suicide_comments_thread_id_set = set(comments['thread_id'])
    submissions['thread_id'] = [x if x in suicide_comments_thread_id_set else None for x in submissions['thread_id']]

    return submissions, comments


def clean_data(submissions, comments, start_year, end_year):
    submissions, comments = process_thread_ids(submissions, comments)

    submissions['created'] = pd.to_datetime(submissions['created'], format='%Y-%m-%d %H:%M')
    submissions_cleaned = submissions.dropna(subset=['thread_id', 'author', 'text'])
    submissions_cleaned = submissions_cleaned[submissions_cleaned['author'] != 'u/[deleted]']
    submissions_cleaned = submissions_cleaned[submissions_cleaned['text'] != '[removed]']
    submissions_filtered = submissions_cleaned[(submissions_cleaned['created'].dt.year >= start_year) & (submissions_cleaned['created'].dt.year <= end_year)]

    submissions_filtered['content'] = submissions_filtered['title'].fillna('') + " " + submissions_filtered['text'].fillna('')
    submissions_final = submissions_filtered[['author', 'score', 'content', 'link', 'thread_id']]


    comments['created'] = pd.to_datetime(comments['created'], format='%Y-%m-%d %H:%M')
    comments_cleaned = comments.dropna(subset=['thread_id', 'author', 'body'])
    comments_cleaned = comments_cleaned[comments_cleaned['author'] != 'u/[deleted]']
    comments_cleaned = comments_cleaned[comments_cleaned['body'] != '[removed]']
    comments_filtered = comments_cleaned[(comments_cleaned['created'].dt.year >= start_year) & (comments_cleaned['created'].dt.year <= end_year)]

    comments_filtered['content'] = comments_filtered['body'].fillna('')
    comments_final = comments_filtered[['author', 'score', 'content', 'link', 'thread_id']]

    return submissions_final, comments_final




# clean data
year = 2023
submissions = pd.read_csv("subreddits_csv/depression_submissions.csv", dtype='str', encoding='utf-8', lineterminator='\n')
comments = pd.read_csv("subreddits_csv/depression_comments.csv", dtype='str', encoding='utf-8', lineterminator='\n')

submissions_final, comments_final = clean_data(submissions, comments, year, year)



subreddit_name = 'depression'
save_dir = f'cleaned_datasets_{year}'
os.makedirs(save_dir, exist_ok=True)
submissions_path = os.path.join(save_dir, f'{subreddit_name}_submissions_cleaned.csv')
comments_path = os.path.join(save_dir, f'{subreddit_name}_comments_cleaned.csv')

submissions_final.to_csv(submissions_path, index=False)
comments_final.to_csv(comments_path, index=False)