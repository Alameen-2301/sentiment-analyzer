from transformers import pipeline

# Load sentiment analysis model
classifier = pipeline("sentiment-analysis")

# Test sentences
sentences = [
    "This movie was absolutely amazing",
    "The food was terrible and cold",
    "I love studying AI",
    "I hate waking up early",
    "The weather today is okay",
    "Thanks for everything god",
    "I am not happy today",
"The movie was not bad at all",
"I am happy but my friend is not"
]

# Run each sentence through the model
for sentence in sentences:
    result = classifier(sentence)
    label = result[0]['label']
    score = round(result[0]['score'] * 100, 2)
    print(f"{sentence}")
    print(f"→ {label} ({score}% confident)")
    print()
    