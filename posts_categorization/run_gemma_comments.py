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

            1. Empathy-based comments:
            - Express understanding and validate feelings (e.g. It’s completely normal to feel this way.)
            - Offer emotional support, comfort, or encouragement (e.g. You’re not alone in this.)

            2. Advice-based comments:
            - Provide specific and practical suggestions (e.g. Have you considered reaching out to a clinical psychologist?)
            - Share resources like hotlines, tools, or articles (e.g. You might find this helpful: [link].)

            3. Comments sharing similar experience:
            - Share personal stories that are similar to the submitter’s situation (e.g. I’ve been through something similar and it was really tough.)
            - Express solidarity and connection (e.g. I know exactly how you feel because I’ve been there too.)

            4. Others/random:
            - Off-topic, humorous, sarcastic, or unclear comments (e.g. LOL, don’t do it! or This reminds me of a movie I watched.)

            5. Invalid:
            - Empty or deleted comment

            DO NOT GIVE ME REASON. ONLY TELL ME THE CATEGORY.

            Your output format must be:
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



