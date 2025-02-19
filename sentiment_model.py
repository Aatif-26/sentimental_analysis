import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv("tweet_emotions.csv")  # Ensure CSV has 'sentence' & 'emotion' columns

# Preprocess Text
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)
    text = " ".join([word for word in tokens if word not in stop_words])
    return text

df['clean_sentence'] = df['content'].apply(clean_text)

# Split Data
X = df['clean_sentence']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert Text to Features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save Model and Vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model Training Completed! 🎉")
