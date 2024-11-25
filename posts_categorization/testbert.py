from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="bert-large-uncased", device=0)

labels = ["seeking help and support", "sharing emotional experience", "giving advice", "others/random"]

posts = [
    "I hate existing. Nothing helps. I think this is my third time posting here and every time I just feel worse. I don't know what to do anymore. Anyone else feel the same?",
    "I'm trying my best to get through this tough time. But I always end up crying when I'm alone. I don't want to burden anyone. I just need to make it through somehow.",
    "Try talking to a professional. It really helped me when I was feeling low. You don’t have to go through this alone.",
    "I’m struggling with my job. It’s been hard to focus on work and even harder to get motivated.",
    "Here’s a random thought I had today while walking down the street."
]

threshold = 0.5

for post in posts:
    result = classifier(post, candidate_labels=labels)

    # Get the highest label and score
    best_label = result['labels'][0]
    best_score = result['scores'][0]

    # Check if the confidence score is high enough to categorize
    if best_score < threshold:
        best_label = "others/random"

    print(f"Post: {post}\nCategory: {best_label} (Confidence: {best_score:.2f})\n")

