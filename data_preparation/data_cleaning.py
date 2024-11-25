import pandas as pd

suicide_submissions = pd.read_csv("subreddits23csv/SuicideWatch_submissions.csv")
suicide_comments = pd.read_csv("subreddits23csv/SuicideWatch_comments.csv")
depression_submissions = pd.read_csv("subreddits23csv/depression_submissions.csv")
depression_comments = pd.read_csv("subreddits23csv/depression_comments.csv")


suicide_submissions['created'] = pd.to_datetime(suicide_submissions['created'], format='%Y-%m-%d %H:%M')
suicide_comments['created'] = pd.to_datetime(suicide_comments['created'], format='%Y-%m-%d %H:%M')
depression_submissions['created'] = pd.to_datetime(depression_submissions['created'], format='%Y-%m-%d %H:%M')
depression_comments['created'] = pd.to_datetime(depression_comments['created'], format='%Y-%m-%d %H:%M')


filtered_suicide_submissions = suicide_submissions[(suicide_submissions['created'].dt.year >= 2012) & (suicide_submissions['created'].dt.year <= 2022)]
filtered_suicide_comments = suicide_comments[(suicide_comments['created'].dt.year >= 2012) & (suicide_comments['created'].dt.year <= 2022)]
filtered_depression_submissions = depression_submissions[(depression_submissions['created'].dt.year >= 2012) & (depression_submissions['created'].dt.year <= 2022)]
filtered_depression_comments = depression_comments[(depression_comments['created'].dt.year >= 2012) & (depression_comments['created'].dt.year <= 2022)]

# # Calculate the exact time range for each filtered dataset
# suicide_submissions_range = (filtered_suicide_submissions['created'].min(), filtered_suicide_submissions['created'].max())
# suicide_comments_range = (filtered_suicide_comments['created'].min(), filtered_suicide_comments['created'].max())
# depression_submissions_range = (filtered_depression_submissions['created'].min(), filtered_depression_submissions['created'].max())
# depression_comments_range = (filtered_depression_comments['created'].min(), filtered_depression_comments['created'].max())
# print("SuicideWatch Submissions Time Range:", suicide_submissions_range)
# print("SuicideWatch Comments Time Range:", suicide_comments_range)
# print("Depression Submissions Time Range:", depression_submissions_range)
# print("Depression Comments Time Range:", depression_comments_range)

filtered_suicide_submissions['content'] = filtered_suicide_submissions['title'].fillna('') + " " + filtered_suicide_submissions['text'].fillna('')
filtered_suicide_comments['content'] = filtered_suicide_comments['body'].fillna('')
filtered_depression_submissions['content'] = filtered_depression_submissions['title'].fillna('') + " " + filtered_depression_submissions['text'].fillna('')
filtered_depression_comments['content'] = filtered_depression_comments['body'].fillna('')

final_suicide_submissions = filtered_suicide_submissions[['author', 'content']]
final_suicide_comments = filtered_suicide_comments[['author', 'content']]
final_depression_submissions = filtered_depression_submissions[['author', 'content']]
final_depression_comments = filtered_depression_comments[['author', 'content']]

final_suicide_submissions.to_csv('SuicideWatch_submissions_cleaned.csv', index = False)
final_suicide_comments.to_csv('SuicideWatch_comments_cleaned.csv', index = False)
final_depression_submissions.to_csv('depression_submissions_cleaned.csv', index = False)
final_depression_comments.to_csv('depression_comments_cleaned.csv', index = False)