import csv

import requests
import json
import pandas as pd
from tqdm import tqdm
import argparse


def ask_gemma2(prompt, port):
    # Define the base URL
    # url = "http://localhost:11435/api/generate"
    url = f"http://localhost:{port}/api/generate"


    # Define the payload
    payload = {
        "model": "gemma2:9b",
        "prompt": prompt,
        "stream": False
    }

    res = []
    # Send POST request
    try:
        response = requests.post(
            url=url,
            data=json.dumps(payload),  # Convert the payload to JSON string
            headers={"Content-Type": "application/json"}  # Specify JSON content type
        )

        # Handle streaming response
        if response.status_code == 200:
            # print("Response from the API:")
            response_json = response.json()
            return response_json['response'].strip()
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")



def generate_prompt(post_content):
    return f"""
            Post: "{post_content}"
            
            
            Categorize the post into one of the categories:
            1.Empathy-based comments: express understanding, support, or compassion.
            2.Advice-based comments: provide suggestions, guidance, or actionable steps.
            3.Comments sharing similar experience: recount the commenter's own experiences that relate to the poster’s situation.
            4.Others/random: do not fit into the above categories.
            5.Invalid:it is empty or deleted
            
            DO NOT GIVE ME REASON. ONLY TELL ME THE CATEGORY.
            
            your output format must be: 
            "1.Empathy-based comments" OR "2.Advice-based comments" OR "3.Comments sharing similar experience" OR "4.Others/random" OR "5.Invalid"
            
            NO REASONING 
            """



if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Categorize comments using Gemma2 API.")
    parser.add_argument("--port", type=int, required=True, help="Port of the API server.")
    parser.add_argument("--input_dataset", type=str, required=True, help="Path to the input dataset (CSV file).")
    parser.add_argument("--output_dataset", type=str, required=True, help="Path to the output dataset (CSV file).")

    args = parser.parse_args()

    # Load input dataset
    input_data = pd.read_csv(args.input_dataset)

    # Prepare output file
    with open(args.output_dataset, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["author", "score", "content", "link", "thread_id", "comments_categ"])

    # Process and categorize comments
    with tqdm(total=len(input_data), desc="Processing Comments", ncols=100) as pbar:
        with open(args.output_dataset, mode="a", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            for index, row in input_data.iterrows():
                # Stop after 100 rows for testing purposes
                # if index >= 100:
                #     break

                content = row['content']
                score = row['score']
                author = row['author']
                link = row['link']
                thread_id = row['thread_id']

                # Categorize the content using the API
                result = ask_gemma2(generate_prompt(content), args.port)
                writer.writerow([author, score, content, link, thread_id, result])
                pbar.set_postfix({"": result[:5]})
                pbar.update(1)






#
# suicide_comments_cleaned = pd.read_csv('cleaned_datasets_2022/SuicideWatch_comments_cleaned.csv')
# depression_comments_cleaned = pd.read_csv('cleaned_datasets_2022/depression_comments_cleaned.csv')
#
#
#
#
#
# with open("suicide_comments_categorized_gemma.csv", mode="w", newline='', encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(["author,content,link,thread_id,submissions_categ"])
#
#
# with tqdm(total=len(suicide_comments_cleaned), desc="SuicideWatch") as pbar:
#     with open("suicide_comments_categorized_gemma.csv", mode="a", newline='', encoding="utf-8") as file:
#         writer = csv.writer(file)
#         for index, row in suicide_comments_cleaned.iterrows():
#             if index > 100:
#                 break
#             content = row['content']
#             author = row['author']
#             link = row['link']
#             thread_id = row['thread_id']
#             result = ask_gemma2(generate_prompt(content))
#             writer.writerow([author, content, link, thread_id, result])
#             pbar.set_postfix({"Last Result": result})
#             pbar.update(1)
#
#
#
# with open("depression_comments_categorized_gemma.csv", mode="w", newline='', encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(["author,content,link,thread_id,submissions_categ"])
#
#
# with tqdm(total=len(depression_comments_cleaned), desc="Depression") as pbar:
#     with open("depression_comments_categorized_gemma.csv", mode="a", newline='', encoding="utf-8") as file:
#         writer = csv.writer(file)
#         for index, row in depression_comments_cleaned.iterrows():
#             if index > 100:
#                 break
#             content = row['content']
#             author = row['author']
#             link = row['link']
#             thread_id = row['thread_id']
#             result = ask_gemma2(generate_prompt(content))
#             writer.writerow([author, content, link, thread_id, result])
#             pbar.set_postfix({"Last Result": result})
#             pbar.update(1)
