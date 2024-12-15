import pandas as pd
import re
import numpy as np

suicide_submissions = pd.read_csv("subreddits_csv/SuicideWatch_submissions.csv", dtype='str', encoding='utf-8', lineterminator='\n')
suicide_comments = pd.read_csv("subreddits_csv/SuicideWatch_comments.csv", dtype='str', encoding='utf-8', lineterminator='\n')
depression_submissions = pd.read_csv("subreddits_csv/depression_submissions.csv", dtype='str', encoding='utf-8', lineterminator='\n')
depression_comments = pd.read_csv("subreddits_csv/depression_comments.csv", dtype='str', encoding='utf-8', lineterminator='\n')

def extract_thread_id(url):
    match = re.search(r'/comments/([a-zA-Z0-9]+)/', url)
    return match.group(1) if match else None

suicide_submissions['thread_id'] = [extract_thread_id(x) for x in suicide_submissions['link']]
suicide_comments['thread_id'] = [extract_thread_id(x) for x in suicide_comments['link']]
depression_submissions['thread_id'] = [extract_thread_id(x) for x in depression_submissions['link']]
depression_comments['thread_id'] = [extract_thread_id(x) for x in depression_comments['link']]

suicide_comments_thread_id_set = set(suicide_comments['thread_id'])
depression_comments_thread_id_set = set(depression_comments['thread_id'])

suicide_submissions['thread_id'] = [x if x in comments_thread_id_set else None for x in subs_thread_id]
depression_submissions['thread_id'] = [x if x in comments_thread_id_set else None for x in subs_thread_id]

suicide_submissions.to_csv('linked_datasets_all/SuicideWatch_linked_submissions.csv', index=False)
suicide_comments.to_csv('linked_datasets_all/SuicideWatch_linked_comments.csv', index=False)
depression_submissions.to_csv('linked_datasets_all/depression_linked_submissions.csv', index=False)
depression_comments.to_csv('linked_datasets_all/depression_linked_comments.csv', index=False)
