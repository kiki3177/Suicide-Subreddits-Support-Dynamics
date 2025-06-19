import pandas as pd

def process_subreddit(subreddit_name, year, core_or_random, num_rows=100):
    original_df = pd.read_csv(f"../pagerank/pagerank_results/{subreddit_name}_pagerank_scores_{year}.csv")

    if core_or_random == 'core':
        target_commenters = original_df.head(num_rows)
    elif core_or_random == 'random':
        target_commenters = original_df.tail(num_rows)
    else:
        raise ValueError("core_or_random must be either 'head' or 'tail'")

    linked_df = pd.read_csv(f"../posts_categorization/linked_submissions_comments/{year}_{subreddit_name}_linked_llama_gemma_qwen.csv", dtype='str', encoding='utf-8', lineterminator='\n')

    result_commenters = linked_df[linked_df['commenter_username'].isin(target_commenters['node'])]
    result_submitters = linked_df[linked_df['submitter_username'].isin(result_commenters['submitter_username'])]
    result_submitters_replied = result_submitters[
        result_submitters['submitter_username'] == result_submitters['commenter_username']]
    result_submitters_replied = result_submitters_replied.drop(result_submitters_replied.columns[0], axis=1)

    result_submitters_replied_agg = result_submitters_replied.groupby('submitter_username', as_index=False).agg({
        'score_submission': 'first',
        'content_submission': 'first',
        'commenter_username': 'first',
        'score_comment': 'first',
        'content_comment': lambda x: ' | '.join(x),
        'thread_id': 'first',
        'url': 'first'
    })

    output_filename = f"target_commenters/{subreddit_name}_{core_or_random}_submitters_replied_{year}.csv"
    result_submitters_replied_agg.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")


subreddit_names = ["suicide", "depression"]
years = [2023, 2024]
core_or_random_list = ['core', 'random']

for subreddit_name in subreddit_names:
    for year in years:
        for core_or_random in core_or_random_list:
            process_subreddit(subreddit_name, year, core_or_random, num_rows=101)



