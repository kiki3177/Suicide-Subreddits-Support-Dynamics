from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="xxxxxxxx",
    base_url="xxxxxxxx"
)
def generate_prompt(post_content):
    return f"""
    If a user were to share the following message, how would you respond?
    Generate 5 different responses and limit each response within 150 words.
    Here is the message: {post_content}
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



