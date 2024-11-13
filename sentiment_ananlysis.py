from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import torch

# Load model and tokenizer
task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Download label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

def sentiment(texts):
    results = {}
    for text in texts:
        text = text[:1500]
        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded_input)
        scores = output[0][0].cpu().numpy()
        scores = softmax(scores)

        # Get the sentiment with the highest score
        max_index = np.argmax(scores)
        results[text] = labels[max_index]
    return results

# Input multiple news articles from the user
user_inputs = []
num_articles = int(input("Enter the number of news articles: "))

for _ in range(num_articles):
    article = input("Please enter the news text: ")
    user_inputs.append(article)

# Get sentiments for the input news articles
sentiment_results = sentiment(user_inputs)

# Display the results
for article, sentiment in sentiment_results.items():
    print(f"The sentiment of the news article is: {sentiment}")

# Save the model and tokenizer
model.save_pretrained("saved_model/sentiment_model")
tokenizer.save_pretrained("saved_model/sentiment_model")

print("Model and tokenizer have been saved.")
