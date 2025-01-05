# IMDB Movie Review Sentiment Analysis App
# Author: Your Name
# Description: A Streamlit application that allows users to input movie reviews 
# and classifies them as positive or negative using a pre-trained sentiment analysis model.
# Meta Title: Sentiment Analysis of IMDB Movie Reviews - Streamlit App
# Meta Description: Explore the power of sentiment analysis with our interactive Streamlit app! 
# Enter any movie review to classify it as positive or negative using a pre-trained model.
# Meta Keywords: Sentiment Analysis, IMDB Reviews, Movie Review Classification, Streamlit App, Natural Language Processing, Machine Learning, Text Analysis

# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
## streamlit app
# Streamlit app

st.set_page_config(
        page_title='IMDB Movie Review Sentiment Analysis',           
        )

st.title('Sentiment Analysis of IMDB Movie Reviews - Streamlit App')
st.write('Explore the power of sentiment analysis with my interactive Streamlit app! Enter any movie review to classify it as positive or negative using a simple recurrent neural network model I created using Tensorflow .')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

