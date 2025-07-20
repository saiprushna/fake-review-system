import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# 1. Load Dataset from CSV
csv_df = pd.read_csv('sample_reviews.csv')
# Add more synthetic reviews to increase dataset size
extra_reviews = [
    {"review_text": "Absolutely love this product, works as expected!", "product": "Speaker", "website": "Amazon", "label": "real"},
    {"review_text": "Terrible quality, stopped working after a day.", "product": "Charger", "website": "Flipkart", "label": "real"},
    {"review_text": "Best deal ever, can't believe the price!", "product": "TV", "website": "AliExpress", "label": "fake"},
    {"review_text": "Received a different item than ordered.", "product": "Watch", "website": "eBay", "label": "fake"},
    {"review_text": "Fast delivery, product as described.", "product": "Mouse", "website": "Amazon", "label": "real"},
    {"review_text": "Fake product, doesn't match the description.", "product": "Shoes", "website": "Flipkart", "label": "fake"},
    {"review_text": "Excellent value for money, highly recommend.", "product": "Tablet", "website": "Amazon", "label": "real"},
    {"review_text": "Scam! Never received my order.", "product": "Camera", "website": "eBay", "label": "fake"},
    {"review_text": "Product works perfectly, very happy.", "product": "Headphones", "website": "Amazon", "label": "real"},
    {"review_text": "Unbelievable offer, buy now!", "product": "Laptop", "website": "AliExpress", "label": "fake"},
    {"review_text": "Arrived on time, good packaging.", "product": "Tablet", "website": "Flipkart", "label": "real"},
    {"review_text": "Worst experience, total waste of money.", "product": "Smartphone", "website": "eBay", "label": "fake"},
    {"review_text": "Very satisfied with the purchase.", "product": "Shoes", "website": "Amazon", "label": "real"},
    {"review_text": "Do not buy this, it's a scam!", "product": "Camera", "website": "AliExpress", "label": "fake"},
    {"review_text": "Great product, will buy again.", "product": "Speaker", "website": "Flipkart", "label": "real"},
    {"review_text": "Received used item instead of new.", "product": "Laptop", "website": "eBay", "label": "fake"},
    {"review_text": "Product exceeded my expectations.", "product": "Tablet", "website": "Amazon", "label": "real"},
    {"review_text": "Item never arrived, lost my money.", "product": "Smartwatch", "website": "AliExpress", "label": "fake"},
    {"review_text": "Very comfortable and fits well.", "product": "Shoes", "website": "Flipkart", "label": "real"},
    {"review_text": "Best purchase I've made this year!", "product": "Headphones", "website": "Amazon", "label": "real"},
    {"review_text": "Fake review, don't trust this seller.", "product": "Tablet", "website": "eBay", "label": "fake"},
]
extra_df = pd.DataFrame(extra_reviews)
df = pd.concat([csv_df, extra_df], ignore_index=True)

# 2. Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['review_text'].apply(preprocess_text)

# 3. Sentiment Analysis
sia = SentimentIntensityAnalyzer()
def get_sentiment(text):
    return sia.polarity_scores(str(text))['compound']

df['sentiment'] = df['review_text'].apply(get_sentiment)

# 4. Feature Extraction (TF-IDF + Sentiment)
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['clean_text'])
X = np.hstack((X_tfidf.toarray(), df[['sentiment']].values))
y = df['label'].map({'real': 0, 'fake': 1}).values

# 5. Train/Test Split (Stratified)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.33, random_state=42, stratify=y)

# 6. Model Training
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 7. Evaluation
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['real', 'fake']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualization
plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['real', 'fake'], yticklabels=['real', 'fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 8. Prediction Function
def predict_review(review):
    clean = preprocess_text(review)
    sent = get_sentiment(review)
    tfidf_vec = tfidf.transform([clean]).toarray()
    features = np.hstack((tfidf_vec, [[sent]]))
    pred = clf.predict(features)[0]
    return 'fake' if pred == 1 else 'real'

# Print predictions for each review in the test set
print("\nPredictions on test set:")
for i in idx_test:
    review = df.loc[i, 'review_text']
    actual = df.loc[i, 'label']
    pred = predict_review(review)
    print(f"Review: {review}\nActual: {actual} | Predicted: {pred}\n")

# Example usage
new_review = "Absolutely fantastic product, will buy again!"
print(f"Review: '{new_review}' => Prediction: {predict_review(new_review)}") 