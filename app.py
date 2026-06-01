"""Streamlit app to detect spam SMS messages using a pre-trained TensorFlow model."""

import os

# Must be set before importing tensorflow to suppress C++ backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import random
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences


# Load model and tokenizer
@st.cache_resource
def load_model():
    """Load the pre-trained Machine Learning model."""
    return tf.keras.models.load_model("./models/model_wordembed.keras")


# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    """Load the pre-trained tokenizer."""
    tok = Tokenizer(num_words=1000, oov_token="<OOV>")
    tok.word_index = np.load(
        "./models/tokenizer_word_index.npy", allow_pickle=True
    ).item()
    return tok


# Load the model and tokenizer
model = load_model()
tokenizer = load_tokenizer()

# Load examples from JSON
with open("examples.json", "r", encoding="utf-8") as file:
    examples_list = json.load(file)


# Preprocessing function
def preprocess_sms(sms_text, maxlen=100):
    """Preprocess the SMS text."""
    sequence = tokenizer.texts_to_sequences([sms_text])
    padded = pad_sequences(sequence, maxlen=maxlen, padding="post", truncating="post")
    return padded


# Prediction function
def predict_sms(sms_text):
    """Predict the label of the SMS message."""
    preprocessed_text = preprocess_sms(sms_text)
    prediction = model.predict(preprocessed_text, verbose=0)[0][0]
    is_spam = prediction > 0.5
    result_label = "Spam" if is_spam else "Non-Spam"
    result_confidence = float(prediction * 100 if is_spam else (1 - prediction) * 100)
    return result_label, result_confidence


# Apply custom CSS
st.markdown(
    """
    <style>
        /* Add background color */
        .stApp {
            background: linear-gradient(to right, #f8f9fa, #e9ecef);
            color: #333333;
        }
        /* Style the title */
        h1 {
            font-size: 3em;
            color: #007BFF;
            text-align: center;
        }
        /* Style the subtitle */
        .subtitle {
            text-align: center;
            font-size: 1.3em;
            color: #555555;
            margin-bottom: 20px;
        }
        /* Style the text input box */
        textarea {
            font-size: 1.2em;
            background-color: #fefefe;
        }
        /* Style the button */
        div.stButton > button {
            background-color: #007BFF;
            color: white;
            font-size: 1.1em;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        div.stButton > button:hover {
            background-color: #0056b3;
        }
        /* Footer styling */
        footer {
            text-align: center;
            font-size: 1em;
            color: #555555;
            margin-top: 50px;
        }
        footer a {
            color: #007BFF;
            text-decoration: none;
        }
        .warning-note {
            background-color: #fff4e5;
            padding: 15px;
            text-align: center;
            border: 1px solid #ffd699;
            border-radius: 5px;
            font-size: 1.1em;
            color: #8a6d3b;
            margin: 20px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App layout
st.title("📲 SMS Spam Detector")
st.markdown(
    "<div class='subtitle'>"
    "Analyze SMS messages and determine whether they're <b>Spam</b> or <b>Non-Spam</b>."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Input section — apply any pending random example before the widget renders
if "_pending_example" in st.session_state:
    st.session_state.user_input_area = st.session_state.pop("_pending_example")
elif "user_input_area" not in st.session_state:
    st.session_state.user_input_area = random.choice(examples_list)

st.markdown("### 📝 Enter your message :")
user_input = st.text_area("Enter your SMS message", height=150, key="user_input_area", label_visibility="collapsed")

col1, col2 = st.columns(2)
with col1:
    predict_button = st.button("🔍 Predict")
with col2:
    if st.button("🎲 Random Example"):
        st.session_state["_pending_example"] = random.choice(examples_list)
        st.rerun()

# Prediction history
if "history" not in st.session_state:
    st.session_state.history = []

if predict_button:
    with st.spinner("Analyzing message..."):
        if len(user_input.strip()) < 5:
            st.error(
                "❌ The message is too short to analyze. Please enter a longer SMS."
            )
        else:
            label, confidence = predict_sms(user_input.strip())
            st.session_state.history.append(
                {
                    "Message": user_input.strip(),
                    "Label": label,
                    "Confidence": f"{confidence:.2f}%",
                }
            )

            EMOJI = "✅" if label == "Non-Spam" else "🚨"
            st.success(f"**Predicted Label:** {EMOJI} {label}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            st.progress(confidence / 100)

            if label == "Spam":
                st.warning("⚠️ **Suspicious SMS detected !**")

# Display prediction history
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Prediction History")
    st.data_editor(pd.DataFrame(st.session_state.history), width="stretch")

# Proportion of Spam vs Non-Spam
if st.session_state.history:
    labels = [h["Label"] for h in st.session_state.history]
    fig = px.pie(
        names=labels,
        title="Classification Results: Spam vs Non-Spam",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.3,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        title_font_size=20,
        title_x=0.25,
        title_y=0.9,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5},
    )
    st.plotly_chart(fig)

st.markdown("---")

# Warning note
st.markdown(
    "<div style='text-align:center;padding:15px;background-color:#fff4e5;border-radius:8px;'>"
    "This application is powered by a Deep Learning model trained to detect spam messages.<br>"
    "Always exercise <b>caution</b> with unsolicited messages."
    "</div>",
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    "<footer style='text-align:center;'>"
    "Made with ❤️ by "
    "<a href='https://www.linkedin.com/in/christophenoret' target='_blank'>"
    "Christophe NORET"
    "</a>"
    "</footer>",
    unsafe_allow_html=True,
)
