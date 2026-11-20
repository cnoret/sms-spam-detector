"""Streamlit app to detect spam SMS messages with a TF-IDF + logistic regression model."""

import json
import random

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📲",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Fitted pipeline and decision threshold, both produced by train.py."""
    return joblib.load("./models/spam_classifier.joblib")


@st.cache_data
def load_metrics():
    with open("./models/metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


bundle = load_model()
pipeline, threshold = bundle["pipeline"], bundle["threshold"]
metrics = load_metrics()

with open("examples.json", "r", encoding="utf-8") as file:
    examples_list = json.load(file)


def predict_sms(sms_text):
    spam_proba = float(pipeline.predict_proba([sms_text])[0, 1])
    is_spam = spam_proba >= threshold
    result_label = "Spam" if is_spam else "Non-Spam"
    return result_label, spam_proba * 100


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    test = metrics["deployed"]["test"]
    cm = test["confusion_matrix"]
    st.markdown("#### 📊 Model Performance")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Spam precision", f"{test['precision']:.1%}")
        st.metric("Spam F1", f"{test['f1']:.2f}")
    with col_b:
        st.metric("Spam recall", f"{test['recall']:.1%}")
        st.metric("Accuracy", f"{test['accuracy']:.1%}")
    st.caption(
        f"{metrics['data']['test']:,} held-out unique messages · "
        f"{cm['fp']} legitimate SMS flagged, {cm['fn']} spams missed"
    )

    st.markdown("---")

    st.markdown("#### 🧠 Model")
    st.markdown(
        f"""
        | Step | Details |
        | --- | --- |
        | Features | TF-IDF · character 2–5-grams |
        | Classifier | Logistic regression (balanced) |
        | Threshold | {threshold:.2f} (precision ≥ {metrics['min_precision_target']:.0%}) |
        """
    )
    if "network" in metrics:
        st.caption(
            f"Beats the original neural network on the same split: "
            f"F1 {test['f1']:.2f} vs {metrics['network']['test']['f1']:.2f}"
        )

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
                label, spam_pct = predict_sms(user_input.strip())
            st.session_state.history.append({
                "Message": user_input.strip(),
                "Label": label,
                "Spam probability": f"{spam_pct:.1f}%",
            })
            st.session_state.last_result = {
                "error": False,
                "label": label,
                "spam_pct": spam_pct,
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
        spam_pct = result["spam_pct"]
        color = "#28a745" if label == "Non-Spam" else "#dc3545"
        emoji = "✅" if label == "Non-Spam" else "🚨"
        st.markdown(
            f"""
            <div style="background:{color};color:white;border-radius:12px;
                        padding:28px 20px;text-align:center;">
                <div style="font-size:2.8em;">{emoji}</div>
                <div style="font-size:1.9em;font-weight:700;margin:10px 0;">{label}</div>
                <div style="font-size:1.1em;margin-bottom:14px;">
                    Spam probability: <b>{spam_pct:.1f}%</b>
                </div>
                <div style="background:rgba(255,255,255,0.3);border-radius:6px;height:8px;">
                    <div style="background:white;border-radius:6px;height:8px;
                                width:{spam_pct:.0f}%;"></div>
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
                "Spam probability": st.column_config.TextColumn(
                    "Spam probability", width="small"
                ),
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
