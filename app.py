from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS
import os

# Load or train your model as before (reuse your code)
# For demo, let's assume you train on sample_reviews.csv as before

app = Flask(__name__)
CORS(app)

# --- Preprocessing setup (reuse your code) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_sentiment(text):
    return sia.polarity_scores(str(text))['compound']

DATASET_FILE = '1429_1 (1).csv'
REVIEW_COL = 'reviews.text'
LABEL_COL = 'reviews.doRecommend'

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

# Load and train model with error handling
if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f'{DATASET_FILE} not found! Please ensure the dataset is present.')
try:
    sample_data = pd.read_csv(DATASET_FILE)
except Exception as e:
    raise RuntimeError(f'Error reading {DATASET_FILE}: {e}')

if REVIEW_COL not in sample_data.columns or LABEL_COL not in sample_data.columns:
    raise ValueError(f'{DATASET_FILE} must have columns: {REVIEW_COL}, {LABEL_COL}')

# Map label: TRUE -> genuine, FALSE/empty -> fake
sample_data[LABEL_COL] = sample_data[LABEL_COL].map(lambda x: 'genuine' if str(x).strip().upper() == 'TRUE' else 'fake')

sample_data['cleaned_text'] = sample_data[REVIEW_COL].astype(str).apply(clean_text)
sample_data['sentiment'] = sample_data[REVIEW_COL].astype(str).apply(get_sentiment)
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(sample_data['cleaned_text'])
X_train = np.hstack((X_train_tfidf.toarray(), sample_data[['sentiment']].values))
y_train = sample_data[LABEL_COL]
model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data.get('review')
    if not review:
        return jsonify({'error': 'No review provided'}), 400
    cleaned = clean_text(review)
    sentiment = get_sentiment(review)
    tfidf_vec = tfidf.transform([cleaned]).toarray()
    features = np.hstack((tfidf_vec, [[sentiment]]))
    pred = model.predict(features)[0]
    label = str(pred).lower()
    return jsonify({'prediction': label})

@app.route('/search_review', methods=['POST'])
def search_review():
    data = request.json
    review = data.get('review')
    if not review:
        return jsonify({'error': 'No review provided'}), 400
    try:
        df = pd.read_csv(DATASET_FILE)
    except Exception as e:
        return jsonify({'error': f'Error reading dataset: {e}'}), 500
    if REVIEW_COL not in df.columns or LABEL_COL not in df.columns:
        return jsonify({'error': 'CSV missing required columns.'}), 500
    # Map label for search as well
    df[LABEL_COL] = df[LABEL_COL].map(lambda x: 'genuine' if str(x).strip().upper() == 'TRUE' else 'fake')
    review_norm = str(review).strip().lower()
    df['review_text_norm'] = df[REVIEW_COL].astype(str).str.strip().str.lower()
    match = df[df['review_text_norm'] == review_norm]
    if not match.empty:
        label = str(match.iloc[0][LABEL_COL]).lower()
        return jsonify({'result': label})
    else:
        return jsonify({'result': 'not found'})

if __name__ == '__main__':
    app.run()  # No debug=True for production 
