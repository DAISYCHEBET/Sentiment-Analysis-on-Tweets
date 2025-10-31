# Sentiment Analysis on Tweets
💬 Tweet Sentiment Analyzer (LSTM + Streamlit)
This project sentiment analysis tool built with TensorFlow/Keras, Streamlit, and a custom NLP preprocessing pipeline. It predicts tweet sentiment—positive, neutral, or negative—using a trained Bidirectional LSTM model and offers both single and batch prediction modes through an intuitive web interface.

🌟 Key Features
Interactive Streamlit App

Predict sentiment for individual tweets with emoji feedback and probability bars

Upload CSV/XLSX files for batch predictions with live streaming and uncertainty filtering

Download results in CSV or Excel format

Custom Preprocessing Pipeline

Handles contractions, slang, stopwords, and lemmatization

Automatically detects preprocessing.py or preprocess.py

Model Highlights

Embedding layer trained from scratch

Bidirectional LSTM with dropout and dense layers

Class weights and early stopping to handle imbalance and overfitting

Project Structure
Code
├── app.py                  # Streamlit app
├── sentiment_lstm.keras   # Trained model
├── tokenizer.pkl          # Fitted tokenizer
├── preprocessing.py       # Text cleaning and lemmatization
├── requirements.txt       # Python dependencies

⚙️ Setup Instructions
Clone the repository

bash
git clone https://github.com/your-username/tweet-sentiment-analyzer.git
cd tweet-sentiment-analyzer
Install dependencies

bash
pip install -r requirements.txt
Run the app

bash
streamlit run app.py
🧠 Model Configuration
Tokenizer: Top 10,000 words, padded to 50 tokens

Embedding: 100-dimensional vectors

LSTM: Bidirectional with 128 units

Loss: sparse_categorical_crossentropy

Optimizer: Adam (learning rate 1e-3)

Regularization: Dropout + class weights

EarlyStopping: Monitors validation loss

🔍 Prediction Modes
1️⃣ Single Tweet
Input a tweet manually or use quick examples

View cleaned text, predicted label, and class probabilities

Visualized with Altair horizontal bar chart

📁 Batch Upload
Upload .csv or .xlsx file with tweet column

Select text column and set uncertainty threshold

Choose between fast vectorized or streamed prediction

Filter and sort results by label or confidence

Download predictions as CSV or Excel

📊 Output Columns
cleaned_text_for_model

pred_label, pred_label_id

prob_positive, prob_neutral, prob_negative

max_prob, uncertain (flagged if confidence is low)



Tweet Sentiment Analysis with LSTM
This project is a complete sentiment analysis pipeline for tweets, built with TensorFlow/Keras, Streamlit, and a custom NLP preprocessing module. It classifies tweets into positive, neutral, or negative sentiments using a trained Bidirectional LSTM model and offers an interactive web interface for both single and batch predictions.

Project Overview
Goal: Predict sentiment from tweets using deep learning and NLP.

Model: Bidirectional LSTM trained on lemmatized, tokenized tweet data.

Interface: Streamlit app for real-time predictions and batch uploads.

Preprocessing: Handles contractions, slang, stopwords, and lemmatization.

Directory Structure
Code
├── app.py                  # Streamlit app
├── sentiment_lstm.keras   # Trained LSTM model
├── tokenizer.pkl          # Tokenizer used during training
├── preprocessing.py       # Text cleaning and lemmatization
├── requirements.txt       # Python dependencies


Setup Instructions:

Clone the repository

install dependencies:
bash
pip install -r requirements.txt

Run the Streamlit app:
bash
streamlit run app.py


Model Details
Tokenizer: Top 10,000 words, padded to 50 tokens

Embedding: 100-dimensional vectors learned during training


Architecture:

Embedding layer

Bidirectional LSTM (128 units)

Dense layers with ReLU and Dropout

Softmax output for 3-class classification

Training Techniques:

Class weights to handle imbalance

EarlyStopping to prevent overfitting

Validation split for monitoring generalization

App Features
Single Tweet Prediction
Input a tweet manually

View cleaned text, predicted sentiment, and confidence scores

Emoji feedback and probability bar chart

Batch Prediction
Upload .csv or .xlsx file with tweets

Select text column and set uncertainty threshold

Stream predictions with progress bar

Filter results and download as CSV or Excel