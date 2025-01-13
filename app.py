""" Streamlit app to detect spam SMS messages using a pre-trained TensorFlow model. """

import os
import random
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Load model and tokenizer
@st.cache_resource
def load_model():
    """Load the pre-trained Machine Learning model."""
    return tf.keras.models.load_model("./models/model_wordembed.keras")


@st.cache_resource
def load_tokenizer():
    """Load the pre-trained tokenizer."""
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.word_index = np.load(
        "./models/tokenizer_word_index.npy", allow_pickle=True
    ).item()
    return tokenizer


model = load_model()
tokenizer = load_tokenizer()


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
    prediction = model.predict(preprocessed_text)[0][0]
    label = "Spam" if prediction > 0.5 else "Non-Spam"
    confidence = prediction * 100 if label == "Spam" else (1 - prediction) * 100
    return label, confidence


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

# Input section
if "user_input" not in st.session_state:
    st.session_state.user_input = (
        "Congratulations! You've won a $1,000 Walmart gift card. "
        "Go to http://bit.ly/123456 to claim now."
    )

st.markdown("### 📝 Enter your message :")
st.text_area("", value=st.session_state.user_input, height=150, key="user_input")

# Prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Predict button and results
if st.button("🔍 Predict"):
    user_input = st.session_state.user_input
    if len(user_input.strip()) < 5:
        st.error("❌ The message is too short to analyze. Please enter a longer SMS.")
    else:
        label, confidence = predict_sms(user_input.strip())
        st.session_state.history.append(
            {
                "Message": user_input.strip(),
                "Label": label,
                "Confidence": f"{confidence:.2f}%",
            }
        )

        # Display prediction
        emoji = "✅" if label == "Non-Spam" else "🚨"
        st.success(f"**Predicted Label:** {emoji} {label}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        if label == "Spam":
            st.warning("⚠️ Be careful! This message looks suspicious.")

# Display prediction history
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Prediction History")
    st.table(pd.DataFrame(st.session_state.history))

# Proportion of Spam vs Non-Spam predictions
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
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig)

st.markdown("---")
st.markdown(
    "<div class='warning-note'>"
    "This application is powered by a Deep Learning model trained to detect spam messages.<br>"
    "Always exercise <b>caution</b> with unsolicited messages."
    "</div>",
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    "<footer>"
    "Made with ❤️ by "
    "<a href='https://www.linkedin.com/in/christophenoret' target='_blank'>"
    "Christophe NORET"
    "</a>"
    "</footer>",
    unsafe_allow_html=True,
)
