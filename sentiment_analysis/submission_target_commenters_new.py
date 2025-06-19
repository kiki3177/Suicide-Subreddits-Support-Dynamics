import pandas as pd

def process_subreddit(subreddit_name, year, core_or_random, num_rows=100):
    original_df = pd.read_csv(f"../pagerank/pagerank_results/{subreddit_name}_pagerank_scores_{year}.csv")

    if core_or_random == 'core':
        target_commenters = original_df.head(num_rows)
    elif core_or_random == 'random':
        target_commenters = original_df.tail(num_rows)
    else:
        raise ValueError("core_or_random must be either 'core' or 'random'")

    linked_df = pd.read_csv(f"../posts_categorization/linked_submissions_comments/{year}_{subreddit_name}_linked_llama_gemma_qwen.csv", dtype='str', encoding='utf-8', lineterminator='\n')

    result_commenters = linked_df[linked_df['commenter_username'].isin(target_commenters['node'])]

    # Filter out comments that are '[deleted]'
    result_commenters_filtered = result_commenters[result_commenters['content_comment'] != '[deleted]']

    result_commenters_filtered = result_commenters_filtered[['submitter_username', 'commenter_username',
                                                            'content_submission', 'content_comment',
                                                            'comments_categ_llama', 'comments_categ_gemma',
                                                            'comments_categ_qwen']]

    output_filename = f"target_commenters_new/{subreddit_name}_{core_or_random}_comments_{year}.csv"
    result_commenters_filtered.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")

subreddit_names = ["suicide", "depression"]
years = [2023, 2024]
core_or_random_list = ['core', 'random']

for subreddit_name in subreddit_names:
    for year in years:
        for core_or_random in core_or_random_list:
            process_subreddit(subreddit_name, year, core_or_random, num_rows=101)
