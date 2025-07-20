# Fake Review Detection System

This project is an AI-powered web application that helps users detect fake product reviews. It uses natural language processing (NLP) and machine learning to classify reviews as real or fake.

## Features
- Paste a single review to instantly analyze if it's real or fake
- Modern, responsive user interface for desktop and mobile
- Uses advanced NLP (TF-IDF, sentiment analysis) and machine learning (Logistic Regression)
- CSV dataset support for model training

## Requirements
- Python 3.7+
- pip (Python package manager)
- See `requirement.txt` for all dependencies

## Installation
1. **Clone or download this repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```
3. **Download NLTK resources (first run will do this automatically):**
   - punkt
   - stopwords
   - wordnet
   - vader_lexicon

## Usage
### 1. Run the backend server
```bash
python app.py
```

### 2. Open the web interface
- Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.
- Paste a review and click "Analyze Review" to see the prediction.

### 3. Model Training
- The model is trained using `fake_review_detector.py` with the dataset in `sample_reviews.csv`.
- You can add more reviews to `sample_reviews.csv` to improve accuracy.

## Customization
- **UI:** Edit `style.css` and `index.html` for look and feel.
- **Model/Data:** Edit `fake_review_detector.py` and `sample_reviews.csv` for model logic and data.

## Credits
- Built with Python, Flask, scikit-learn, NLTK, and modern web technologies.
- UI inspired by neon and glassmorphism design trends.

---

*Made with ❤️ by [Your Name]* 