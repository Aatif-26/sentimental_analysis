import pickle
import re
import nltk
from nltk.tokenize import word_tokenize

# Load Model & Vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)
    text = " ".join(tokens)
    return text

# Get user input and make prediction
if __name__ == "__main__":
    while True:
        sentence = input("Enter a sentence (or type 'exit' to quit): ").strip()
        if sentence.lower() == 'exit':
            print("Exiting...")
            break
        
        sentence_cleaned = clean_text(sentence)
        sentence_tfidf = vectorizer.transform([sentence_cleaned])
        prediction = model.predict(sentence_tfidf)[0]

        print(f"Predicted Emotion: {prediction}\n")
