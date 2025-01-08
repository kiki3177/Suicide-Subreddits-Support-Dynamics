import pandas as pd
import re

def extract_category(text):
    text = str(text)
    match = re.findall(r'\d', text)
    if len(match) == 1:
        return int(match[0])
    else:
        return text


def process_llm_data(llm, year, subreddit_name):
    dataset_submissions = pd.read_csv(f"llm_results/categorization_results_{llm}_{year}/{year}_{subreddit_name}_submissions_categorized_{llm}.csv", dtype='str', encoding='utf-8', lineterminator='\n')
    dataset_comments = pd.read_csv(f"llm_results/categorization_results_{llm}_{year}/{year}_{subreddit_name}_comments_categorized_{llm}.csv", dtype='str', encoding='utf-8', lineterminator='\n')

    dataset_submissions = dataset_submissions.dropna(subset=['thread_id'])
    dataset_comments = dataset_comments.dropna(subset=['thread_id'])

    linked_data = pd.merge(dataset_submissions, dataset_comments, on="thread_id", how='inner')


    linked_data[f'submissions_categ'] = linked_data['submissions_categ'].apply(extract_category)
    linked_data[f'comments_categ'] = linked_data['comments_categ'].apply(extract_category)

    linked_data[f'submissions_categ'] = linked_data[f'submissions_categ'].apply(
        lambda x: '4' if str(x) not in ['1', '2', '3'] else x
    )
    linked_data[f'comments_categ'] = linked_data[f'comments_categ'].apply(
        lambda x: '4' if str(x) not in ['1', '2', '3'] else x
    )

    linked_data.rename(columns={
        "submissions_categ": f"submissions_categ_{llm}",
        "comments_categ": f"comments_categ_{llm}"
    }, inplace=True)

    linked_data = linked_data[
        ['author_x', 'score_x', 'content_x', f'submissions_categ_{llm}', 'author_y', 'score_y', 'content_y',
         f'comments_categ_{llm}', 'thread_id', 'link_x']]
    linked_data.rename(columns={
        "author_x": "submitter_username",
        "score_x": "score_submission",
        "content_x": "content_submission",
        "author_y": "commenter_username",
        "score_y": "score_comment",
        "content_y": "content_comment",
        "link_x": "url"
    }, inplace=True)

    return linked_data




year = 2023
subreddit_name = 'depression'
llms = ['llama', 'gemma', 'qwen']
all_data = []

for llm in llms:
    llm_data = process_llm_data(llm, year, subreddit_name)
    all_data.append(llm_data)

final_data = all_data[0]

for other_data in all_data[1:]:
    final_data = final_data.merge(other_data, on=['submitter_username', 'score_submission', 'content_submission',
                                                  'commenter_username', 'score_comment', 'content_comment', 'thread_id',
                                                  'url'], how='inner')


final_data = final_data.drop_duplicates(subset=['submitter_username', 'score_submission', 'content_submission',
                                                'commenter_username', 'score_comment', 'content_comment',
                                                'thread_id', 'url'])
final_data = final_data.dropna()

final_data = final_data[
    ['submitter_username', 'score_submission', 'content_submission',
     'submissions_categ_llama', 'submissions_categ_gemma', 'submissions_categ_qwen',
     'commenter_username', 'score_comment', 'content_comment',
     'comments_categ_llama', 'comments_categ_gemma', 'comments_categ_qwen',
     'thread_id', 'url']
]

final_data.to_csv(f"linked_submissions_comments/{year}_{subreddit_name}_linked_{llms[0]}_{llms[1]}_{llms[2]}.csv")


print(final_data.isna().sum())
print(final_data.shape)