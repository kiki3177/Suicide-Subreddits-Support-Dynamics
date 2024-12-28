import praw
import pandas as pd
import flask
from flask import Flask, request
app = Flask(__name__)

@app.route('/reddit_callback')
def reddit_callback():
    # Retrieve the authorization code or access token from the URL parameters
    authorization_code = request.args.get('code')
    # Do something with the authorization code, such as exchanging it for an access token
    # Or, store it for later use
    return "Callback received successfully"

# if __name__ == '__main__':
#     app.run(host='localhost', port=8088)


reddit = praw.Reddit(
    client_id='RsbqDq0SXaDSniDUBryiPg',
    client_secret='f2AjugZMbLtvWHrzxl-rnjBqVpS4eQ',
    user_agent='personal_script by /u/Taotao0711',
    check_for_async=False
)


subreddit = reddit.subreddit('SuicideWatch')

titles = []
bodies = []
urls = []
comments_list = []

# Iterate over the posts
# create a tqdm and update manually

# for submission in subreddit.new(limit=10000):  # Adjust the limit as needed
from tqdm import tqdm
td = tqdm(subreddit.new(limit=1500), total=1500)
for submission in td:
    submission.comments.replace_more(limit=0)  # Remove MoreComments
    comments = submission.comments.list()[:10]  # Get top 10 comments

    # Append data to lists
    titles.append(submission.title)
    bodies.append(submission.selftext)
    urls.append(submission.url)
    comments_list.append([comment.body for comment in comments])

# Creating a DataFrame
df = pd.DataFrame({
    'Title': titles,
    'Body': bodies,
    'URL': urls,
    'Top Comments': comments_list
})

print(df)
# Saving the df
# df.to_csv('Reddit.csv', index=False)