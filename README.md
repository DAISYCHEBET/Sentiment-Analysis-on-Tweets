# Sentiment Analysis on Tweets
Overview

This project is a sentiment analysis system built to classify tweets as positive, negative, or neutral. Using deep learning techniques with LSTM (Long Short-Term Memory) networks, the model analyzes textual data and predicts the underlying sentiment. The project also includes a Streamlit web app for both individual tweet sentiment prediction and batch analysis of multiple tweets.


Features

Data Preprocessing:

Text cleaning, tokenization, and padding for consistent input shapes.
Handling of emojis, hashtags, and special characters.


Model:

LSTM neural network for sequential text analysis.
Trained on labeled tweet data for sentiment classification.
Saves the trained model for reuse in the Streamlit app.



Streamlit Web App:

Individual Prediction: Input a single tweet to get sentiment probability and label.
Batch Prediction: Upload a CSV or Excel file of tweets for sentiment analysis of multiple entries.
Visualizations: Sentiment distribution charts for uploaded datasets.



Installation

Clone the repository


Install dependencies:
pip install -r requirements.txt


Run the Streamlit app:
bash
streamlit run app.py




Usage
Individual Tweet Prediction


Open the Streamlit app.
Go to the Individual Prediction section.
Input a single tweet and click Predict.
View the sentiment label and probability.


Batch Prediction

Go to the Batch Prediction section.
Upload a CSV or Excel file containing tweets.
The app will process all tweets and show a sentiment distribution chart.
Download the prediction results if needed.


Project Structure
├── app.py                  # Streamlit application
├── preprocessing.py        # Text preprocessing functions
├── model/                  # Saved trained LSTM model
├── data/                   # Sample datasets
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation



Dependencies

Python 3.10+
TensorFlow / Keras
Streamlit
Pandas, Numpy
Joblib
Matplotlib/Seaborn (for visualizations)




Future Improvements

Improve model performance using regularization 

Experiment with transformer-based models (e.g., BERT) for improved accuracy.

Add multilingual support for tweets in other languages.


Author:
Daisy  Chebet
