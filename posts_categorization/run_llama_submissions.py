import csv

import requests
import json
import pandas as pd
from tqdm import tqdm
import argparse


def ask_llama3(prompt, port):
    # Define the base URL
    # url = "http://localhost:11435/api/generate"
    url = f"http://localhost:{port}/api/generate"

    # Define the payload
    payload = {
        "model": "llama3:8b",
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

            1. Seeking help and support:
            - Express emotional distress (e.g. I feel like I’m falling apart and I don’t know who to talk to.)
            - Explicitly request support or advice (e.g. Does anyone have advice on how to cope with these thoughts?)

            2. Sharing experience:
            - Describe personal struggles or mental health challenges (e.g. Here’s my story from when I hit rock bottom last year.)
            - Reflect on past events related to mental health (e.g. It’s been a year since my attempt, and I wanted to share what I’ve learned.)

            3. Giving advice:
            - Offer personal strategies or suggestions (e.g. What helped me was keeping a routine, even on the hardest days.)
            - Recommend resources or professional help (e.g. You might find this helpful: [link].)

            4. Others/random:
            - Off-topic, humorous, sarcastic, or unclear posts (e.g. LOL, it is so fun! or What’s your favorite TV show right now?)

            5. Invalid:
            - Empty or deleted post

            DO NOT GIVE ME REASON. ONLY TELL ME THE CATEGORY.

            Your output format must be:
            "1.Seeking help and support" OR "2.Sharing experience" OR "3.Giving advice" OR "4.Others/random" OR "5.Invalid"

            NO REASONING 
            """


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Categorize submissions using Llama3 API.")
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
                result = ask_llama3(generate_prompt(content), args.port)
                writer.writerow([author, score, content, link, thread_id, result])
                pbar.set_postfix({"": result[:5]})
                pbar.update(1)
