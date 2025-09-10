📌 Emotion Detection from Text

📖 Project Overview

This project focuses on detecting emotions expressed in text data using Natural Language Processing (NLP) and Machine Learning techniques. The dataset contains sentences labeled with emotions such as anger, fear, joy, love, sadness, and surprise.

The notebook demonstrates the complete pipeline:

  a. Text preprocessing (cleaning, tokenization, stopword removal, lemmatization)
  
  b. Feature extraction using Count Vectorizer and TF-IDF
  
  c. Training an XGBoost Classifier
  
  d. Evaluating the model with accuracy, precision, recall, F1-score, and confusion matrix

📊 Dataset

  > File: text_emotions.csv

  > Features:
        a. content → text message/tweet

        b. sentiment → emotion label (anger, fear, joy, love, sadness, surprise)

  > The dataset is used to train and test the emotion detection model.

🔧 Methodology

1. Data Exploration
   
      a. Checked dataset size, info, class distribution.

      b. Visualized sentiment counts with Seaborn.

3. Text Preprocessing
   
  a. Removed punctuations and numbers.
  
  b. Tokenized text into words.
  
  c. Removed stopwords (extended with domain-specific words).
  
  d. Lemmatized tokens using WordNet Lemmatizer.

5. Feature Extraction
   
  a. Count Vectorizer → converts text into a word-count matrix.
  
  b. TF-IDF Vectorizer → assigns importance weights to words.

7. Model Training
   
  a.Used XGBoost Classifier with tuned parameters (max_depth=16, n_estimators=1000).
  
  b.Encoded target labels using LabelEncoder.

9. Evaluation
    
  a. Accuracy, Precision, Recall, F1-score (macro average).
  
  b. Confusion Matrix.
  
  c. Classification Report with all emotion classes.

✅ Results

1. The SVM model achieved strong performance across all metrics.

2. Certain emotions like joy and sadness were classified with higher accuracy compared to others like fear and love.

3. TF-IDF features generally improved results over raw counts.

🔮 Future Improvements

1. Use deep learning models (LSTM, BERT, DistilBERT) for better context understanding.

2. Apply data augmentation to balance class distribution.

3. Deploy the model as an API or web app for real-time emotion detection.
