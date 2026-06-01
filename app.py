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


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./models/model_wordembed.keras")


@st.cache_resource
def load_tokenizer():
    tok = Tokenizer(num_words=1000, oov_token="<OOV>")
    tok.word_index = np.load(
        "./models/tokenizer_word_index.npy", allow_pickle=True
    ).item()
    return tok


model = load_model()
tokenizer = load_tokenizer()

with open("examples.json", "r", encoding="utf-8") as file:
    examples_list = json.load(file)


def preprocess_sms(sms_text, maxlen=100):
    sequence = tokenizer.texts_to_sequences([sms_text])
    return pad_sequences(sequence, maxlen=maxlen, padding="post", truncating="post")


def predict_sms(sms_text):
    prediction = model.predict(preprocess_sms(sms_text), verbose=0)[0][0]
    is_spam = prediction > 0.5
    label = "Spam" if is_spam else "Non-Spam"
    confidence = float(prediction * 100 if is_spam else (1 - prediction) * 100)
    return label, confidence


st.markdown(
    """
    <style>
        footer { visibility: hidden; }
        .custom-footer {
            text-align: center;
            font-size: 1em;
            color: #555555;
            margin-top: 40px;
        }
        .custom-footer a { color: #007BFF; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📲 SMS Spam Detector")
st.markdown(
    "<p style='text-align:center;font-size:1.2em;color:#555555;margin-bottom:1.5rem;'>"
    "Analyze SMS messages and determine whether they're <b>Spam</b> or <b>Non-Spam</b>."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "_pending_example" in st.session_state:
    st.session_state.user_input_area = st.session_state.pop("_pending_example")
elif "user_input_area" not in st.session_state:
    st.session_state.user_input_area = random.choice(examples_list)

# ── Two-column layout: input (left) | result (right) ─────────────────────────
col_input, col_result = st.columns([3, 2])

with col_input:
    st.markdown("### 📝 Enter your message")
    user_input = st.text_area(
        "Enter your SMS message",
        height=160,
        key="user_input_area",
        label_visibility="collapsed",
    )
    btn_left, btn_right = st.columns(2)
    with btn_left:
        predict_button = st.button("🔍 Predict", type="primary", use_container_width=True)
    with btn_right:
        if st.button("🎲 Random", use_container_width=True):
            st.session_state["_pending_example"] = random.choice(examples_list)
            st.rerun()

with col_result:
    st.markdown("### 📊 Result")

    if predict_button:
        if len(user_input.strip()) < 5:
            st.session_state.last_result = {"error": True}
        else:
            with st.spinner("Analyzing..."):
                label, confidence = predict_sms(user_input.strip())
            st.session_state.history.append({
                "Message": user_input.strip(),
                "Label": label,
                "Confidence": f"{confidence:.2f}%",
            })
            st.session_state.last_result = {
                "error": False,
                "label": label,
                "confidence": confidence,
            }

    result = st.session_state.last_result
    if result is None:
        st.markdown(
            "<div style='text-align:center;color:#aaa;padding:48px 0;font-size:1em;'>"
            "Run a prediction to see results here."
            "</div>",
            unsafe_allow_html=True,
        )
    elif result["error"]:
        st.error("❌ Message too short. Please enter a longer SMS.")
    else:
        label = result["label"]
        confidence = result["confidence"]
        color = "#28a745" if label == "Non-Spam" else "#dc3545"
        emoji = "✅" if label == "Non-Spam" else "🚨"
        st.markdown(
            f"""
            <div style="background:{color};color:white;border-radius:12px;
                        padding:28px 20px;text-align:center;">
                <div style="font-size:2.8em;">{emoji}</div>
                <div style="font-size:1.9em;font-weight:700;margin:10px 0;">{label}</div>
                <div style="font-size:1.1em;margin-bottom:14px;">
                    Confidence: <b>{confidence:.1f}%</b>
                </div>
                <div style="background:rgba(255,255,255,0.3);border-radius:6px;height:8px;">
                    <div style="background:white;border-radius:6px;height:8px;
                                width:{confidence:.0f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── History + Chart ───────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Prediction History")
    st.data_editor(pd.DataFrame(st.session_state.history), width="stretch")

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

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;padding:15px;background-color:#fff4e5;"
    "border-radius:8px;color:#8a6d3b;'>"
    "This application is powered by a Deep Learning model trained to detect spam messages.<br>"
    "Always exercise <b>caution</b> with unsolicited messages."
    "</div>",
    unsafe_allow_html=True,
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='custom-footer'>Made with ❤️ by "
    "<a href='https://www.linkedin.com/in/christophenoret' target='_blank'>"
    "Christophe NORET</a></div>",
    unsafe_allow_html=True,
)
