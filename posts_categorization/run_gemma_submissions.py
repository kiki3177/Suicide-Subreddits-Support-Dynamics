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
            1.Seeking help and support: expresses a need for help or support.
            2.Sharing experience: shares their personal experiences or stories.
            3.Giving advice: provides advice or suggestions to others.
            4.Others/random: do not fit into the above categories.
            5.Invalid:it is empty or deleted

            DO NOT GIVE ME REASON. ONLY TELL ME THE CATEGORY.

            your output format must be: 
            "1.Seeking help and support" OR "2.Sharing experience" OR "3.Giving advice" OR "4.Others/random" OR "5.Invalid"

            NO REASONING 
            """


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Categorize submissions using Gemma2 API.")
    parser.add_argument("--port", type=int, required=True, help="Port of the API server.")
    parser.add_argument("--input_dataset", type=str, required=True, help="Path to the input dataset (CSV file).")
    parser.add_argument("--output_dataset", type=str, required=True, help="Path to the output dataset (CSV file).")

    args = parser.parse_args()

    # Load input dataset
    input_data = pd.read_csv(args.input_dataset)

    # Prepare output file
    with open(args.output_dataset, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["author", "score", "content", "link", "thread_id", "submissions_categ"])

    # Process and categorize submissions
    with tqdm(total=len(input_data), desc="Processing Submissions", ncols=100) as pbar:
        with open(args.output_dataset, mode="a", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            for index, row in input_data.iterrows():
                # # Stop after 100 rows for testing purposes
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
