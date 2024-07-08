import streamlit as st
import numpy as np
import string
import joblib
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Load models and preprocessors
glove_model = joblib.load('glove_model.pkl')
svd_model = joblib.load('svd_model.pkl')
scaler = joblib.load('scaler.pkl')
restored_model = load_model('my_model.h5')

# Load NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def document_vector(words):
    valid_words = [word for word in words if word in glove_model]
    if not valid_words:
        return np.zeros(glove_model.vector_size)
    return np.mean(glove_model[valid_words], axis=0)


def adding_stemming(words):
    return [stemmer.stem(word) for word in words]

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = adding_stemming(text)
    text = ' '.join(text)
    return text

def predict(text):
    processed_words = preprocess_text(text)
    doc_vec = document_vector(processed_words.split())
    doc_vec = doc_vec.reshape(1, -1)
    reduced_vec = svd_model.transform(doc_vec)
    scaled_vec = scaler.transform(reduced_vec)
    prediction = restored_model.predict(scaled_vec)
    return prediction

# Streamlit application layout
st.title('Dynamic prediction App')
user_input = st.text_area("Enter your text here", "Type Here")

if st.button('Predict'):
    result = predict(user_input)
    if result < 0:
        st.write(f'Prediction: {-1 * result}')
    else:
        st.write(f'Prediction: {result}')
