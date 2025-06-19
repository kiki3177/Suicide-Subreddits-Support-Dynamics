import pandas as pd
from bert_score import score

csv_file_path = "top5_comments_suicide.csv"
data = pd.read_csv(csv_file_path)

submissions = data['content_submission'].tolist()
human_comments = data['content_comment'].tolist()
llm_comments = data['llm_comments'].tolist()

# Compute BERT scores for human comments
P_human, R_human, F1_human = score(human_comments, submissions, lang="en", verbose=True)

# Compute BERT scores for LLM comments
P_llm, R_llm, F1_llm = score(llm_comments, submissions, lang="en", verbose=True)

data['human_P'] = P_human.tolist()
data['human_R'] = R_human.tolist()
data['human_F1'] = F1_human.tolist()

data['llm_P'] = P_llm.tolist()
data['llm_R'] = R_llm.tolist()
data['llm_F1'] = F1_llm.tolist()

average_human_P = data['human_P'].mean()
average_human_R = data['human_R'].mean()
average_human_F1 = data['human_F1'].mean()
average_llm_P = data['llm_P'].mean()
average_llm_R = data['llm_R'].mean()
average_llm_F1 = data['llm_F1'].mean()

print("\nAverage BERT Scores:")
print(f"Human Comments - Precision: {average_human_P:.4f}, Recall: {average_human_R:.4f}, F1: {average_human_F1:.4f}")
print(f"LLM Comments   - Precision: {average_llm_P:.4f}, Recall: {average_llm_R:.4f}, F1: {average_llm_F1:.4f}")

output_file_path = "top5_comments_suicide_bert.csv"
data.to_csv(output_file_path, index=False)

print(f"\nSaved output with BERT scores to {output_file_path}")