import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./models/model_wordembed.keras")


model = load_model()

# Load tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.word_index = np.load(
    "./models/tokenizer_word_index.npy", allow_pickle=True
).item()


# Define preprocessing function
def preprocess_sms(sms_text, maxlen=100):
    # Tokenize and pad the input SMS
    sequence = tokenizer.texts_to_sequences([sms_text])
    padded = pad_sequences(sequence, maxlen=maxlen, padding="post", truncating="post")
    return padded


# Define prediction function
def predict_sms(sms_text):
    preprocessed_text = preprocess_sms(sms_text)
    prediction = model.predict(preprocessed_text)[0][0]  # Get the first prediction
    label = "Spam" if prediction > 0.5 else "Non-Spam"
    confidence = prediction * 100 if label == "Spam" else (1 - prediction) * 100
    return label, confidence


# Streamlit App Interface
st.title("SMS Spam Detector")
st.write("Enter an SMS message below to classify it as Spam or Non-Spam.")

# Text input
DEFAULT_SMS = "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now."
user_input = st.text_area("Enter SMS text:", value=DEFAULT_SMS, height=100)

if st.button("Predict"):
    if user_input.strip():
        label, confidence = predict_sms(user_input.strip())
        st.write(f"**Predicted Label:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    else:
        st.write("Please enter a valid SMS message.")
