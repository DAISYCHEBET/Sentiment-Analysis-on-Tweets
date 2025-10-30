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
