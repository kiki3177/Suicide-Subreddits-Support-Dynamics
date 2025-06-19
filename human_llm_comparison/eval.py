import pandas as pd
from tqdm import tqdm
from compute_metrics import RewardFunction


# put your csv file here:
csv_path = "top5_comments_suicide_results.csv"

df = pd.read_csv(csv_path)


# select the metric you want to use: "rouge_l", "bleu_score", and "bertscore"
metric = "bertscore"

# iterate over each dataframe to calculate metric
metric_column = []

computeMetric = RewardFunction()

overall_score = 0.0
length = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Metric Calculation"):
    if row["content_comment"] == "[deleted]" or pd.isna(row["llm_comments"]) or pd.isna(row["content_comment"]):
        metric_column.append(0.0)
        continue
    tempScore = computeMetric.compute_reward(str(row["llm_comments"]), str(row["content_comment"]), metric)
    metric_column.append(tempScore)
    overall_score += tempScore
    length += 1

df[metric] = metric_column
df.to_csv("top5_comments_suicide_results.csv", index = False)

print(f"Overall {metric}: {overall_score / length}")

