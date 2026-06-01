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

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📲",
    layout="wide",
)


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
    result_label = "Spam" if is_spam else "Non-Spam"
    result_confidence = float(prediction * 100 if is_spam else (1 - prediction) * 100)
    return result_label, result_confidence


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### 📊 Model Performance")
    st.metric("Test Accuracy", "97%")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Spam F1", "0.89")
    with col_b:
        st.metric("Ham F1", "0.98")
    st.caption("1,115 held-out messages · UCI SMS Spam Collection")

    st.markdown("---")

    st.markdown("#### 🧠 Architecture")
    st.markdown(
        """
        | Layer | Details |
        | --- | --- |
        | Embedding | dim 128 · vocab 1K |
        | Pooling | GlobalAveragePooling1D |
        | Dense × 2 | 128 → 64 · ReLU |
        | Output | sigmoid |
        """
    )
    st.caption("153K parameters · 598 KB")

    st.markdown("---")

    st.info(
        "Trained on ~5,500 messages (UCI dataset, 2010s, English only). "
        "For educational purposes - not production-ready.",
        icon="⚠️",
    )

    st.markdown("---")

    st.markdown(
        "<div style='text-align:center;'>"
        "<p style='color:#888;font-size:0.85em;margin-bottom:8px;'>Made by</p>"
        "<a href='https://www.linkedin.com/in/christophenoret/' target='_blank'>"
        "<img src='https://img.shields.io/badge/Christophe_Noret-0A66C2?logo=linkedin&logoColor=white'/>"
        "</a></div>",
        unsafe_allow_html=True,
    )

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        footer { visibility: hidden; }
        h1 { text-align: center; }
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
    st.markdown("### 🎯 Result")

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

    col_table, col_chart = st.columns([3, 2])

    with col_table:
        title_col, clear_col = st.columns([4, 1])
        with title_col:
            st.markdown("### 📜 Prediction History")
        with clear_col:
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.history = []
                st.session_state.last_result = None
                st.rerun()
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            hide_index=True,
            column_config={
                "Message": st.column_config.TextColumn("Message", width="large"),
                "Label": st.column_config.TextColumn("Label", width="small"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            },
        )

    with col_chart:
        labels = [h["Label"] for h in st.session_state.history]
        fig = px.pie(
            names=labels,
            color=labels,
            color_discrete_map={"Spam": "#dc3545", "Non-Spam": "#28a745"},
            hole=0.4,
        )
        fig.update_traces(textinfo="percent", textposition="inside")
        fig.update_layout(
            margin={"t": 30, "b": 30, "l": 20, "r": 20},
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.15,
                "xanchor": "center",
                "x": 0.5,
            },
            height=300,
        )
        st.plotly_chart(fig, width="stretch")
