import requests
import json
import pandas as pd
from tqdm import tqdm

def ask_gemma2(prompt):
    # Define the base URL
    url = "http://localhost:11434/api/generate"

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
Categorize the post into one of the categories:
1.Seeking help and support: expresses a need for help or support.
2.Sharing experience: shares their personal experiences or stories.
3.Giving advice: provides advice or suggestions to others.
4.Others/random: do not fit into the above categories.

ONLY TELL ME THE CATEGORY NUMBER AND NAME, NOT THE REASONING.

Post: "{post_content}"
Category:
"""



suicide_submissions_cleaned = pd.read_csv('../data_preparation/cleaned_datasets/SuicideWatch_submissions_cleaned.csv')
depression_submissions_cleaned = pd.read_csv('../data_preparation/cleaned_datasets/depression_submissions_cleaned.csv')

suicide_submissions_cleaned['submission_categ'] = [ask_gemma2(generate_prompt(content)) for content in tqdm(suicide_submissions_cleaned['content'], desc="SuicideWatch")]
depression_submissions_cleaned['submission_categ'] = [ask_gemma2(generate_prompt(content)) for content in tqdm(depression_submissions_cleaned['content'], desc="SuicideWatch")]



suicide_submissions_cleaned.to_csv("suicide_submissions_categorized.csv", index= False)
depression_submissions_cleaned.to_csv("depression_submissions_categorized.csv", index= False)



# contents = [
#     "I hate existing. Nothing helps. I think this is my third time posting here and every time I just feel worse. I don't know what to do anymore. Anyone else feel the same?",
#     "I'm trying my best to get through this tough time. But I always end up crying when I'm alone. I don't want to burden anyone. I just need to make it through somehow.",
#     "Try talking to a professional. It really helped me when I was feeling low. You don’t have to go through this alone.",
#     "I’m struggling with my job. It’s been hard to focus on work and even harder to get motivated.",
#     "Here’s a random thought I had today while walking down the street."
# ]
# for content in contents:
#     prompt = generate_prompt(content)
#     res = ask_gemma2(prompt)
#     print(res)