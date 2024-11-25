from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="xxxxxxxx",
    base_url="xxxxxxxx"
)
def generate_prompt(post_content):
    return f"""
Categorize the post into one of the categories:
1.Seeking help and support: expresses a need for help or support.
2.Sharing experience: shares their personal experiences or stories.
3.Giving advice: provides advice or suggestions to others.
4.Others/random: do not fit into the above categories.

Post: "{post_content}"
Category:
"""


# non-streaming response
def gpt_35_api(messages: list):

    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    print(completion.choices[0].message.content)

# streaming response
def gpt_35_api_stream(messages: list):
    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

if __name__ == '__main__':

    contents = [
        "I hate existing. Nothing helps. I think this is my third time posting here and every time I just feel worse. I don't know what to do anymore. Anyone else feel the same?",
        "I'm trying my best to get through this tough time. But I always end up crying when I'm alone. I don't want to burden anyone. I just need to make it through somehow.",
        "Try talking to a professional. It really helped me when I was feeling low. You don’t have to go through this alone.",
        "I’m struggling with my job. It’s been hard to focus on work and even harder to get motivated.",
        "Here’s a random thought I had today while walking down the street."
    ]
    for content in contents:
        prompt = generate_prompt(content)
        gpt_35_api([{'role': 'user','content':prompt},])


