import pandas as pd


def extract_top5_threads_and_comments(subreddit_label):
    filepath = f"../posts_categorization/concatenated_linked_datasets/concatenated_{subreddit_label}_linked.csv"
    df = pd.read_csv(filepath)
    df['score_submission'] = pd.to_numeric(df['score_submission'], errors='coerce')
    df['score_comment'] = pd.to_numeric(df['score_comment'], errors='coerce')
    df = df.dropna(subset=['score_submission', 'score_comment'])
    df = df[(df['score_submission'] > 0) & (df['score_comment'] > 0)]

    valid_threads = df.groupby('thread_id').filter(lambda g: len(g) >= 5)

    top_threads = (
        valid_threads.drop_duplicates('thread_id')
        .nlargest(30, 'score_submission')
        .thread_id
    )

    filtered_df = df[df['thread_id'].isin(top_threads)]

    top5_comments = (
        filtered_df.groupby('thread_id', group_keys=False)
        .apply(lambda group: group.nlargest(5, 'score_comment'))
        .reset_index(drop=True)
    )

    assert top5_comments.shape[0] == 150, "Final dataset does not contain exactly 150 comments."

    top5_comments.to_csv(f"top5_human_comments_{subreddit_label}.csv", index=False)

    return top5_comments



extract_top5_threads_and_comments(subreddit_label="suicide")



