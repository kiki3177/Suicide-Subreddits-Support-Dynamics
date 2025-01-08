import pandas as pd

suicide_core_2023 = pd.read_csv("results/suicide_pagerank_scores_2023.csv")
suicide_core_2023 = suicide_core_2023.head(101)


suicide_linked_2023 = pd.read_csv("../posts_categorization/linked_submissions_comments/2023_suicide_linked_llama_gemma_qwen.csv", dtype='str', encoding='utf-8',lineterminator='\n')


result_commenters = suicide_linked_2023[suicide_linked_2023['commenter_username'].isin(suicide_core_2023['node'])]

result_submitters = suicide_linked_2023[suicide_linked_2023['submitter_username'].isin(result_commenters['submitter_username'])]

result_submitters_replied = result_submitters[result_submitters['submitter_username'] == result_submitters['commenter_username']]
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

result_submitters_replied_agg.to_csv("suicide_submitters_replied_2023.csv", index=False)

