import pandas as pd


def clean_data_submissions(df, start_year, end_year):
    df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M')
    df_cleaned = df.dropna(subset=['thread_id', 'author', 'text'])
    df_cleaned = df_cleaned[df_cleaned['author'] != 'u/[deleted]']
    df_cleaned = df_cleaned[df_cleaned['text'] != '[removed]']
    df_filtered = df_cleaned[(df_cleaned['created'].dt.year >= start_year) & (df_cleaned['created'].dt.year <= end_year)]

    return df_filtered

def clean_data_comments(df, start_year, end_year):
    df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M')
    df_cleaned = df.dropna(subset=['thread_id', 'author', 'body'])
    df_cleaned = df_cleaned[df_cleaned['author'] != 'u/[deleted]']
    df_cleaned = df_cleaned[df_cleaned['body'] != '[removed]']
    df_filtered = df_cleaned[(df_cleaned['created'].dt.year >= start_year) & (df_cleaned['created'].dt.year <= end_year)]

    return df_filtered

def integrate_content_submissions(df):
    df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
    df = df[['author', 'score', 'content', 'link', 'thread_id']]
    return df

def integrate_content_comments(df):
    df['content'] = df['body'].fillna('')
    df = df[['author', 'score', 'content', 'link', 'thread_id']]
    return df



suicide_submissions = pd.read_csv("linked_datasets_all/SuicideWatch_linked_submissions.csv", dtype='str', encoding='utf-8',lineterminator='\n')
suicide_comments = pd.read_csv("linked_datasets_all/SuicideWatch_linked_comments.csv", dtype='str', encoding='utf-8', lineterminator='\n')
depression_submissions = pd.read_csv("linked_datasets_all/depression_linked_submissions.csv", dtype='str', encoding='utf-8', lineterminator='\n')
depression_comments = pd.read_csv("linked_datasets_all/depression_linked_comments.csv", dtype='str', encoding='utf-8', lineterminator='\n')



filtered_suicide_submissions_2022 = clean_data_submissions(suicide_submissions, 2022, 2022)
filtered_suicide_comments_2022 =  clean_data_comments(suicide_comments, 2022, 2022)
filtered_depression_submissions_2022 = clean_data_submissions(depression_submissions, 2022, 2022)
filtered_depression_comments_2022 = clean_data_comments(depression_comments, 2022, 2022)

filtered_suicide_submissions_2023 = clean_data_submissions(suicide_submissions, 2023, 2023)
filtered_suicide_comments_2023 =  clean_data_comments(suicide_comments, 2023, 2023)
filtered_depression_submissions_2023 = clean_data_submissions(depression_submissions, 2023, 2023)
filtered_depression_comments_2023 = clean_data_comments(depression_comments, 2023, 2023)





cleaned_suicide_submissions_2022 = integrate_content_submissions(filtered_suicide_submissions_2022)
cleaned_suicide_comments_2022 = integrate_content_comments(filtered_suicide_comments_2022)
cleaned_depression_submissions_2022 = integrate_content_submissions(filtered_depression_submissions_2022)
cleaned_depression_comments_2022 = integrate_content_comments(filtered_depression_comments_2022)

cleaned_suicide_submissions_2023 = integrate_content_submissions(filtered_suicide_submissions_2023)
cleaned_suicide_comments_2023 = integrate_content_comments(filtered_suicide_comments_2023)
cleaned_depression_submissions_2023 = integrate_content_submissions(filtered_depression_submissions_2023)
cleaned_depression_comments_2023 = integrate_content_comments(filtered_depression_comments_2023)




cleaned_suicide_submissions_2022.to_csv("cleaned_datasets_2022/suicide_submissions_cleaned.csv", index = False)
cleaned_suicide_comments_2022.to_csv("cleaned_datasets_2022/suicide_comments_cleaned.csv", index = False)
cleaned_depression_submissions_2022.to_csv("cleaned_datasets_2022/depression_submissions_cleaned.csv", index = False)
cleaned_depression_comments_2022.to_csv("cleaned_datasets_2022/depression_comments_cleaned.csv", index = False)

cleaned_suicide_submissions_2023.to_csv("cleaned_datasets_2023/suicide_submissions_cleaned.csv", index = False)
cleaned_suicide_comments_2023.to_csv("cleaned_datasets_2023/suicide_comments_cleaned.csv", index = False)
cleaned_depression_submissions_2023.to_csv("cleaned_datasets_2023/depression_submissions_cleaned.csv", index = False)
cleaned_depression_comments_2023.to_csv("cleaned_datasets_2023/depression_comments_cleaned.csv", index = False)






filtered_suicide_submissions_all = clean_data_submissions(suicide_submissions, 2022, 2023)
filtered_suicide_comments_all = clean_data_comments(suicide_comments, 2022, 2023)
filtered_depression_submissions_all = clean_data_submissions(depression_submissions, 2022, 2023)
filtered_depression_comments_all = clean_data_comments(depression_comments, 2022, 2023)


def link_datasets_by_thread_id(dataset_submissions, dataset_comments, thread_id_col='thread_id'):
    if thread_id_col not in dataset_submissions.columns or thread_id_col not in dataset_comments.columns:
        raise ValueError(f"Column '{thread_id_col}' must exist in both datasets.")

    linked_data = pd.merge(dataset_submissions, dataset_comments, on=thread_id_col, how='inner')
    linked_data = linked_data[['author_x', 'score_x', 'title', 'text', 'author_y', 'score_y', 'body', 'thread_id', 'url']]
    linked_data.rename(columns={
        "author_x": "author_submission",
        "score_x": "score_submission",
        "title": "title_submission",
        "text": "content_submission",
        "author_y": "author_comment",
        "score_y": "score_comment",
        "body": "content_comment"
    }, inplace=True)

    return linked_data




linked_suicide = link_datasets_by_thread_id(filtered_suicide_submissions_all, filtered_suicide_comments_all)
linked_depression = link_datasets_by_thread_id(filtered_depression_submissions_all, filtered_depression_comments_all)

linked_suicide.to_csv("linked_datasets_2022_2023/2022_2023_linked_suicide.csv")
linked_depression.to_csv("linked_datasets_2022_2023/2022_2023_linked_depression.csv")