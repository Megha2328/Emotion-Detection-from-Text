📌 Emotion Detection from Text

💡 Project Overview

This project focuses on detecting emotions such as anger, fear, joy, love, sadness, and surprise from text data using Natural Language Processing (NLP) and Machine Learning. The idea is to take raw text, clean and process it, convert it into meaningful numerical features, and then train multiple machine learning models to predict the emotion behind the text. The notebook demonstrates the complete workflow, starting from text preprocessing to model training, evaluation, and finally an interactive prediction system where the user can enter their own text and get the predicted emotion in real-time.

📊 Dataset

The dataset used in this project contains short pieces of text (such as sentences or tweets) along with their associated emotion labels. Each text entry is labeled as one of the six possible emotions: anger, fear, joy, love, sadness, or surprise. This dataset is used for both training and evaluating the machine learning models.

⚙️ Requirements

The project requires Python along with some common libraries for machine learning and text processing. You will need to install packages such as numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, and nltk. Additionally, some NLTK resources such as stopwords and WordNet need to be downloaded for text preprocessing.

🚀 How to Run

To run the project, first place the dataset (text_emotions.csv) in the same directory as the notebook. Open the notebook using Jupyter and execute the cells step by step. The notebook will guide you through data preprocessing, feature extraction with CountVectorizer and TF-IDF, training multiple machine learning models, and evaluating their performance. At the end of the notebook, you will find an interactive loop where you can type in your own sentences, and the system will predict the emotion using different models such as SVM, Logistic Regression, Decision Tree, Naive Bayes, XGBoost, and Random Forest.

For example, if you enter:

I am feeling very happy today!


the system may respond with predictions like:

SVM: joy  
Logistic Regression: joy  
Decision Tree: joy  
Naive Bayes: joy  
XGBoost: joy  
Random Forest: joy  

🔧 Models Used

The project explores several classical machine learning models to detect emotions from text. These include Support Vector Machine (SVM), Logistic Regression, Decision Tree, Naive Bayes, XGBoost, and Random Forest. Each model is trained on the processed dataset and evaluated using various metrics to compare their effectiveness.

📈 Evaluation

The models are evaluated using accuracy, precision, recall, and F1-score to get a balanced view of their performance. A confusion matrix and classification report are also generated to better understand how well each model performs for different emotion classes. The experiments show that using TF-IDF features generally results in better performance compared to simple word counts.

🔮 Future Improvements

Although the current models perform well, there is room for improvement. In the future, deep learning methods such as LSTM or transformer-based models like BERT could be incorporated for capturing more context. Data augmentation techniques may also help balance the dataset, especially if certain emotions are underrepresented. Finally, the system could be deployed as a web app or an API for real-world usage, and an ensemble approach that combines predictions from all models could be added for even better accuracy.
