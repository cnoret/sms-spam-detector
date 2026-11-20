"""Streamlit app to detect spam SMS with a neural network, compared with a baseline.

Both models are trained and evaluated by train.py under the same protocol:
- the neural network (word embeddings + character CNN, Keras) is the model studied,
- TF-IDF + logistic regression is the comparison baseline.
"""

import json
import os
import random

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📲",
    layout="wide",
)

MODELS = {
    "network": {
        "label": "🧠 Neural network",
        "short": "🧠 Network",
        "features": "Word embeddings + character CNN",
        "classifier": "Keras · Dense + sigmoid",
    },
    "baseline": {
        "label": "📏 Baseline",
        "short": "📏 Baseline",
        "features": "TF-IDF · character 2–5-grams",
        "classifier": "Logistic regression (balanced)",
    },
}


@st.cache_resource
def load_network():
    """Keras model with its text vectorization inside: it takes the raw message."""
    import keras

    return keras.models.load_model("./models/spam_network.keras")


@st.cache_resource
def load_baseline():
    """Fitted scikit-learn pipeline, produced by train.py."""
    return joblib.load("./models/spam_classifier.joblib")["pipeline"]


@st.cache_data
def load_metrics():
    with open("./models/metrics.json", encoding="utf-8") as f:
        return json.load(f)


metrics = load_metrics()
network, baseline = load_network(), load_baseline()

with open("examples.json", encoding="utf-8") as file:
    examples_list = json.load(file)


def spam_probability(model_key, sms_text):
    if model_key == "network":
        import tensorflow as tf

        return float(network(tf.constant([[sms_text]]), training=False)[0, 0])
    return float(baseline.predict_proba([sms_text])[0, 1])


def verdict(model_key, proba):
    return "Spam" if proba >= metrics[model_key]["threshold"] else "Non-Spam"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    selected = st.session_state.get("model_choice", "network")

    st.markdown("#### 📊 Model Performance")
    rows = {
        "Spam precision": lambda t: f"{t['precision']:.1%}",
        "Spam recall": lambda t: f"{t['recall']:.1%}",
        "Spam F1": lambda t: f"{t['f1']:.3f}",
        "Legit flagged": lambda t: str(t["confusion_matrix"]["fp"]),
        "Spams missed": lambda t: str(t["confusion_matrix"]["fn"]),
    }
    header = " | ".join(MODELS[k]["short"] for k in MODELS)
    table = [f"| | {header} |", "| --- | --- | --- |"]
    for name, fmt in rows.items():
        table.append(f"| {name} | " + " | ".join(fmt(metrics[k]["test"]) for k in MODELS) + " |")
    st.markdown("\n".join(table))
    st.caption(
        f"Same split for both: {metrics['data']['test']:,} held-out unique messages, "
        f"threshold set for precision ≥ {metrics['min_precision_target']:.0%} "
        "on out-of-fold predictions of the train set."
    )
    if "original_network" in metrics:
        st.caption(
            f"First version of the network (Embedding + Dense, spaCy cleaning): "
            f"spam F1 {metrics['original_network']['test']['f1']:.3f} on the same test set."
        )

    st.markdown("---")

    st.markdown(f"#### Selected model: {MODELS[selected]['label']}")
    st.markdown(
        f"""
        | Step | Details |
        | --- | --- |
        | Features | {MODELS[selected]['features']} |
        | Classifier | {MODELS[selected]['classifier']} |
        | Threshold | {metrics[selected]['threshold']:.2f} |
        """
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
    st.radio(
        "Model",
        options=list(MODELS),
        format_func=lambda k: MODELS[k]["label"],
        key="model_choice",
        horizontal=True,
    )
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
            message = user_input.strip()
            with st.spinner("Analyzing..."):
                probas = {k: spam_probability(k, message) for k in MODELS}
            choice = st.session_state.model_choice
            st.session_state.history.append({
                "Message": message,
                "Model": MODELS[choice]["label"],
                "Label": verdict(choice, probas[choice]),
                **{MODELS[k]["label"]: f"{p:.1%}" for k, p in probas.items()},
            })
            st.session_state.last_result = {"error": False, "probas": probas}

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
        choice = st.session_state.model_choice
        spam_pct = result["probas"][choice] * 100
        label = verdict(choice, result["probas"][choice])
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
        st.markdown("##### ⚖️ Both models on this message")
        for col, key in zip(st.columns(2), MODELS, strict=True):
            proba = result["probas"][key]
            with col:
                st.metric(f"{MODELS[key]['label']} · {verdict(key, proba)}", f"{proba:.1%}")

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
                "Model": st.column_config.TextColumn("Model", width="small"),
                "Label": st.column_config.TextColumn("Label", width="small"),
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
