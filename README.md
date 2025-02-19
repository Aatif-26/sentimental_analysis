# Sentiment Analysis Model - Setup & Testing Guide

## Prerequisites
Make sure you have **Python** installed and set up in your system.

## Step 1: Create and Activate Virtual Environment
```bash
python -m venv venv
```
On **Windows**:
```bash
venv\Scripts\activate
```
On **Mac/Linux**:
```bash
source venv/bin/activate
```

## Step 2: Install Required Libraries
```bash
pip install nltk pandas scikit-learn
```

## Step 3: Download Necessary NLTK Data
Run the following commands in a Python script or interactive shell:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```



## Step 4: Run the Sentiment Analysis Model
Run this script once to train the model and save it.:
```bash
python sentiment_model.py
```

## Step 5: Troubleshooting
- If you face issues related to missing **punkt_tab**, delete and re-download **punkt**:
  ```python
  nltk.download('punkt')
  ```
## Step 6: Run test model
  ```bash
  python test_model.py
  ```


## Notes
- Make sure you are running commands inside the **virtual environment**.
- If you still face errors, check that the **content column** exists in your dataset.

Now you're all set to test your sentiment analysis model! 🚀

