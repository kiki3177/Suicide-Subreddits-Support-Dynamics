import praw
import pandas as pd
from datetime import datetime
import flask
from flask import Flask, request
from tqdm import tqdm

app = Flask(__name__)

@app.route('/reddit_callback')

def reddit_callback():
    authorization_code = request.args.get('code')
    return "Callback received successfully."

def scrape_subreddit(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)

    submissions_dict = {
        "author": [],
        "title": [],
        "score": [],
        "created": [],
        "link": [],
        "text": [],
        "url": []
    }
    comments_dict = {
        "author": [],
        "score": [],
        "created": [],
        "link": [],
        "body": []
    }

    td = tqdm(subreddit.top(time_filter='year', limit=1000), total=1000, desc=f"Scraping r/{subreddit_name}")

    for submission in td:
        submission_author_prefix = f"u/{submission.author}" if submission.author else "[deleted]"
        submissions_dict["author"].append(submission_author_prefix)
        submissions_dict["title"].append(submission.title)
        submissions_dict["score"].append(submission.score)
        submission_created_time = datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M')
        submissions_dict["created"].append(submission_created_time)
        submissions_dict["link"].append(submission.url)
        submissions_dict["text"].append(submission.selftext)
        submissions_dict["url"].append(submission.url)

        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            comment_author_prefix = f"u/{comment.author}" if comment.author else "[deleted]"
            comments_dict["author"].append(comment_author_prefix)
            comments_dict["score"].append(comment.score)
            comment_created_time = datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M')
            comments_dict["created"].append(comment_created_time)
            comment_link = f'https://www.reddit.com{comment.permalink}'
            comments_dict["link"].append(comment_link)
            comments_dict["body"].append(comment.body)

    submissions_path = f"subreddits_csv/{subreddit_name}_submissions_2024.csv"
    comments_path = f"subreddits_csv/{subreddit_name}_comments_2024.csv"

    data_submissions = pd.DataFrame(submissions_dict)
    data_submissions = data_submissions.reset_index(drop=True)
    data_submissions.to_csv(submissions_path, index=False)

    data_comments = pd.DataFrame(comments_dict)
    data_comments = data_comments.reset_index(drop=True)
    data_comments.to_csv(comments_path, index=False)

    print(f"Data saved to {submissions_path} and {comments_path}.")




if __name__ == "__main__":
    reddit = praw.Reddit(
        client_id="your_client_id", # replace with your client id
        client_secret="your_client_secret", # replace with your client secret
        user_agent="your_user_agent", # replace with your user agent
        check_for_async=False
    )

    scrape_subreddit("depression")

