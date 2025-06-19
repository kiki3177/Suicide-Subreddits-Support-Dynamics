import csv

import requests
import json
import pandas as pd
from tqdm import tqdm
import argparse


def ask_qwen2(prompt, port):
    # Define the base URL
    # url = "http://localhost:11435/api/generate"
    url = f"http://localhost:{port}/api/generate"


    # Define the payload
    payload = {
        "model": "qwen2:7b",
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

            DO NOT GIVE ME REASON. ONLY TELL ME THE CATEGORY.

            Your output format must be:
            1 OR 2 OR 3 OR 4

            NO REASONING 
            """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorize comments using Qwen2 API.")
    parser.add_argument("--port", type=int, required=True, help="Port of the API server.")
    parser.add_argument("--input_dataset", type=str, required=True, help="Path to the input dataset (CSV file).")
    parser.add_argument("--output_dataset", type=str, required=True, help="Path to the output dataset (CSV file).")

    args = parser.parse_args()

    # Load input dataset
    input_data = pd.read_csv(args.input_dataset)

    # Prepare list to store categorization results
    categ_results = []

    # Process each comment
    with tqdm(total=len(input_data), desc="Categorizing Comments", ncols=100) as pbar:
        for index, row in input_data.iterrows():
            comment_content = row['llm_comments']

            # If comment_content is NaN, skip or label invalid
            if pd.isna(comment_content) or str(comment_content).strip() == "":
                categ_results.append("5")
            else:
                prompt = generate_prompt(comment_content)
                result = ask_qwen2(prompt, args.port)

                if result is None or result == "":
                    categ_results.append("5")
                else:
                    categ_results.append(result)

            pbar.set_postfix({"Last result": categ_results[-1][:10]})
            pbar.update(1)

    # Add new column to dataframe
    input_data['llm_comments_categ_qwen'] = categ_results

    # Write full dataframe with new column to output
    input_data.to_csv(args.output_dataset, index=False, encoding='utf-8')


