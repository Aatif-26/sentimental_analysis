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

## Step 4: Verify NLTK Data Path
To check where NLTK data is stored, run:
```python
import nltk
print(nltk.data.path)
```
If necessary, manually set the correct path:
```python
nltk.data.path.append("C:\\Users\\YourUsername\\AppData\\Roaming\\nltk_data")
```

## Step 5: Run the Sentiment Analysis Model
Execute the script to test the model:
```bash
python sentiment_model.py
```

## Step 6: Troubleshooting
- If you face issues related to missing **punkt_tab**, delete and re-download **punkt**:
  ```python
  nltk.download('punkt')
  ```
- If issues persist, reinstall **NLTK**:
  ```bash
  pip uninstall nltk
  pip install nltk
  ```

## Notes
- Make sure you are running commands inside the **virtual environment**.
- If you still face errors, check that the **content column** exists in your dataset.

Now you're all set to test your sentiment analysis model! 🚀

