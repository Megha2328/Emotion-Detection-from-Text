📌 Emotion Detection from Text
📖 Project Overview

This project focuses on detecting emotions expressed in text data using Natural Language Processing (NLP) and Machine Learning techniques. The dataset contains sentences labeled with emotions such as anger, fear, joy, love, sadness, and surprise.

The notebook demonstrates the complete pipeline:

Text preprocessing (cleaning, tokenization, stopword removal, lemmatization)

Feature extraction using Count Vectorizer and TF-IDF

Training an XGBoost Classifier

Evaluating the model with accuracy, precision, recall, F1-score, and confusion matrix

📊 Dataset

File: text_emotions.csv

Features:

content → text message/tweet

sentiment → emotion label (anger, fear, joy, love, sadness, surprise)

The dataset is used to train and test the emotion detection model.

🔧 Methodology

1. Data Exploration

Checked dataset size, info, class distribution.

Visualized sentiment counts with Seaborn.

2. Text Preprocessing

Removed punctuations and numbers.

Tokenized text into words.

Removed stopwords (extended with domain-specific words).

Lemmatized tokens using WordNet Lemmatizer.

3. Feature Extraction

Count Vectorizer → converts text into a word-count matrix.

TF-IDF Vectorizer → assigns importance weights to words.

4. Model Training

Used XGBoost Classifier with tuned parameters (max_depth=16, n_estimators=1000).

Encoded target labels using LabelEncoder.

5. Evaluation

Accuracy, Precision, Recall, F1-score (macro average).

Confusion Matrix.

Classification Report with all emotion classes.

✅ Results

1. The SVM model achieved strong performance across all metrics.

2. Certain emotions like joy and sadness were classified with higher accuracy compared to others like fear and love.

3. TF-IDF features generally improved results over raw counts.

🔮 Future Improvements

1. Use deep learning models (LSTM, BERT, DistilBERT) for better context understanding.

2. Apply data augmentation to balance class distribution.

3. Deploy the model as an API or web app for real-time emotion detection.
