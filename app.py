import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# Load model and tokenizer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./models/model_wordembed.keras")


model = load_model()

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.word_index = np.load(
    "./models/tokenizer_word_index.npy", allow_pickle=True
).item()


# Preprocessing function
def preprocess_sms(sms_text, maxlen=100):
    sequence = tokenizer.texts_to_sequences([sms_text])
    padded = pad_sequences(sequence, maxlen=maxlen, padding="post", truncating="post")
    return padded


# Prediction function
def predict_sms(sms_text):
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
    </style>
    """,
    unsafe_allow_html=True,
)

# App layout
st.title("📲 SMS Spam Detector")
st.markdown(
    "Analyze SMS messages and determine whether they're **Spam** or **Non-Spam**."
)
st.markdown("---")

# Input section
DEFAULT_SMS = "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now."
st.markdown("### 📝 Enter your message :")
user_input = st.text_area("", value=DEFAULT_SMS, height=150)

# Predict button and results
if st.button("🔍 Predict"):
    if user_input.strip():
        label, confidence = predict_sms(user_input.strip())
        st.success(f"**Predicted Label:** {label}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        if label == "Spam":
            st.warning("⚠️ Be careful! This message looks suspicious.")
    else:
        st.error("❌ Please enter a valid SMS message.")

st.markdown("---")
st.markdown(
    "📖 **Note:** This application is powered by a TensorFlow model trained to detect spam messages. Always exercise caution with unsolicited messages."
)
