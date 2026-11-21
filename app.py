"""Streamlit app: detect spam SMS with a neural network, compared with a baseline.

Both models are trained and evaluated by train.py under the same protocol:
- the neural network (word embeddings + character CNN, Keras) is the model studied,
- TF-IDF + logistic regression is the comparison baseline.

The models read English SMS.
"""

import html
import json
import math
import os
import random

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon=":material/sms:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ACCENT = "#3B4BC8"
SPAM_COLOR = "#B42318"
HAM_COLOR = "#1D4E89"
MUTED = "#4B5563"

MODELS = {
    "network": {
        "label": "Neural network",
        "short": "Network",
        "by": "according to the neural network",
        "features": "Word embeddings + character CNN",
        "classifier": "Keras · dense layer + sigmoid",
    },
    "baseline": {
        "label": "TF-IDF baseline",
        "short": "Baseline",
        "by": "according to the TF-IDF baseline",
        "features": "TF-IDF · character 2–5-grams",
        "classifier": "Logistic regression (balanced classes)",
    },
}

EXAMPLES = {
    "Prize scam": "WINNER!! You have won a £900 prize. Call 09061701461 to claim now",
    "Bank phishing": (
        "Alert! Your bank account has suspicious activity. "
        "Click to verify: http://bank.fakealert.com"
    ),
    "Personal": "Are we still meeting for lunch at noon tomorrow?",
}

RANDOM_EXAMPLE = ":material/shuffle: Random"
MIN_LENGTH = 5
MAX_EXPLAINED_WORDS = 60
REPO_URL = "https://github.com/cnoret/sms-spam-detector"
LINKEDIN_URL = "https://www.linkedin.com/in/christophenoret/"


# ── Models ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading the neural network…")
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


@st.cache_data
def load_examples():
    with open("examples.json", encoding="utf-8") as f:
        return json.load(f)


metrics = load_metrics()
network, baseline = load_network(), load_baseline()


def spam_probabilities(model_key, messages):
    """Spam probability of each message, in one call to the model."""
    if model_key == "network":
        import tensorflow as tf

        batch = tf.constant(list(messages))[:, None]
        return [float(p) for p in network(batch, training=False)[:, 0]]
    return [float(p) for p in baseline.predict_proba(list(messages))[:, 1]]


def verdict(model_key, proba):
    return "Spam" if proba >= metrics[model_key]["threshold"] else "Legit"


def _logit(p, eps=1e-7):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def word_effects(model_key, message):
    """Occlusion: drop each word in turn and measure the change in log-odds.

    A positive effect means the word pushes the message towards spam.
    Every variant goes through the model in a single batch.
    """
    words = message.split()[:MAX_EXPLAINED_WORDS]
    variants = [" ".join(words[:i] + words[i + 1 :]) for i in range(len(words))]
    probas = spam_probabilities(model_key, [message, *variants])
    full = _logit(probas[0])
    return [(w, full - _logit(p)) for w, p in zip(words, probas[1:], strict=True)]


# ── Formatting ────────────────────────────────────────────────────────────────
def fmt_num(x, digits=3):
    return f"{x:.{digits}f}"


def fmt_pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


def fmt_int(n):
    return f"{n:,}"


def fmt_proba(p):
    """Probability as a percentage, without claiming 0% or 100%."""
    if p >= 0.999:
        return "> 99.9%"
    if p < 0.001:
        return "< 0.1%"
    return fmt_pct(p)


def plural(n, singular, plural_form):
    return f"{fmt_int(n)} {singular if n == 1 else plural_form}"


# ── Session state ─────────────────────────────────────────────────────────────
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_result", None)
if "user_input_area" not in st.session_state:
    st.session_state.user_input_area = random.choice(load_examples())


def use_example():
    """Put the picked example in the text area, then release the pill."""
    pick = st.session_state.example_pick
    if pick == RANDOM_EXAMPLE:
        st.session_state.user_input_area = random.choice(load_examples())
    elif pick in EXAMPLES:
        st.session_state.user_input_area = EXAMPLES[pick]
    st.session_state.example_pick = None


def clear_history():
    st.session_state.history = []
    st.session_state.last_result = None


def selected_model():
    return st.session_state.get("model_choice") or "network"


# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
      footer {{ visibility: hidden; }}
      .block-container {{ padding-top: 2.2rem; max-width: 1200px; }}
      .app-header {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
                     margin-bottom: 0.4rem; }}
      .app-logo {{ width: 36px; height: 36px; border-radius: 9px; background: {ACCENT};
                   display: flex; align-items: center; justify-content: center; }}
      .app-title {{ font-size: 1.45rem; font-weight: 700; color: #111827; }}
      .app-badge {{ font-size: 0.75rem; font-weight: 500; color: #374151;
                    background: #F3F4F6; border: 1px solid #E5E7EB; border-radius: 999px;
                    padding: 2px 10px; }}
      .app-subtitle {{ color: {MUTED}; margin: 0 0 1rem 0; }}
      .verdict {{ display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }}
      .verdict-icon {{ width: 48px; height: 48px; border-radius: 12px; display: flex;
                       align-items: center; justify-content: center; flex: none; }}
      .verdict-label {{ font-size: 1.6rem; font-weight: 700; line-height: 1.1; }}
      .muted {{ color: {MUTED}; font-size: 0.85rem; }}
      .proba-row {{ display: flex; justify-content: space-between; font-size: 0.85rem;
                    margin-bottom: 6px; }}
      .proba-track {{ position: relative; height: 10px; background: #EEF0F3;
                      border-radius: 999px; }}
      .proba-fill {{ height: 10px; border-radius: 999px; }}
      .proba-threshold {{ position: absolute; top: -5px; width: 2px; height: 20px;
                          background: #111827; }}
      .mono {{ font-variant-numeric: tabular-nums; font-weight: 500; }}
      .compare {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px;
                  font-size: 0.9rem; margin-top: 6px; }}
      .section-sep {{ border-top: 1px solid #E5E7EB; margin: 16px 0 12px; }}
      .words {{ font-size: 1rem; line-height: 2.1; }}
      .words mark {{ color: #111827; border-radius: 4px; padding: 2px 4px; }}
      .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px;
                     margin: 0 4px 0 10px; vertical-align: middle; }}
      .placeholder {{ color: {MUTED}; text-align: center; padding: 56px 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

LOGO = (
    '<svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" '
    'stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>'
    '<path d="M9 10l2 2 4-4"></path></svg>'
)
ICON_SPAM = (
    f'<svg aria-hidden="true" width="24" height="24" viewBox="0 0 24 24" fill="none" '
    f'stroke="{SPAM_COLOR}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 9v4"></path><path d="M12 17h.01"></path>'
    '<path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z">'
    "</path></svg>"
)
ICON_HAM = (
    f'<svg aria-hidden="true" width="24" height="24" viewBox="0 0 24 24" fill="none" '
    f'stroke="{HAM_COLOR}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20 6L9 17l-5-5"></path></svg>'
)

# ── Header ────────────────────────────────────────────────────────────────────
network_test = metrics["network"]["test"]
baseline_test = metrics["baseline"]["test"]

st.markdown(
    f"""
    <div class="app-header">
      <div class="app-logo">{LOGO}</div>
      <h1 class="app-title" style="margin:0;padding:0;font-size:1.45rem">SMS Spam Detector</h1>
      <span class="app-badge">Keras network · words + characters</span>
    </div>
    <p class="app-subtitle">A neural network trained on the UCI SMS Spam Collection,
    compared with a TF-IDF baseline on the same test set.</p>
    """,
    unsafe_allow_html=True,
)

kpis = [
    (
        "Spam F1 · network",
        fmt_num(network_test["f1"]),
        f"baseline {fmt_num(baseline_test['f1'])}"
        + (
            f" · v1 {fmt_num(metrics['original_network']['test']['f1'])}"
            if "original_network" in metrics
            else ""
        ),
    ),
    (
        "Spam precision",
        fmt_pct(network_test["precision"]),
        plural(
            network_test["confusion_matrix"]["fp"],
            "legit SMS wrongly flagged",
            "legit SMS wrongly flagged",
        ),
    ),
    (
        "Spam recall",
        fmt_pct(network_test["recall"]),
        f"{network_test['confusion_matrix']['fn']} spams missed out of "
        f"{network_test['confusion_matrix']['fn'] + network_test['confusion_matrix']['tp']}",
    ),
    ("Test set", fmt_int(metrics["data"]["test"]), "unique messages, unseen in training"),
]
for col, (label, value, note) in zip(st.columns(4), kpis, strict=True):
    with col.container(border=True):
        st.metric(label, value)
        st.caption(note)

tab_analyse, tab_perf, tab_method = st.tabs(
    [":material/search: Analyze", ":material/monitoring: Performance", ":material/school: Method"]
)


# ── Analyze ───────────────────────────────────────────────────────────────────
def proba_bar(model_key, proba):
    threshold = metrics[model_key]["threshold"]
    color = SPAM_COLOR if proba >= threshold else HAM_COLOR
    return f"""
      <div class="proba-row"><span class="muted">Spam probability</span>
        <span class="mono">{fmt_proba(proba)}</span></div>
      <div class="proba-track">
        <div class="proba-fill" style="width:{proba * 100:.1f}%;background:{color}"></div>
        <div class="proba-threshold" style="left:{threshold * 100:.1f}%"></div>
      </div>
      <div class="muted" style="margin-top:6px">Black mark: decision threshold
        {fmt_num(threshold)}, set to target {fmt_pct(metrics["min_precision_target"], 0)}
        precision.</div>
    """


def highlighted_words(effects):
    strongest = max((abs(e) for _, e in effects), default=0.0)
    parts = []
    for word, effect in effects:
        text = html.escape(word)
        share = abs(effect) / strongest if strongest > 0.05 else 0.0
        if share < 0.15:
            parts.append(text)
            continue
        rgb = "180,35,24" if effect > 0 else "29,78,137"
        alpha = 0.12 + 0.38 * share
        parts.append(f'<mark style="background:rgba({rgb},{alpha:.2f})">{text}</mark>')
    return " ".join(parts)


def render_result(result):
    model_key = selected_model()
    proba = result["probas"][model_key]
    label = verdict(model_key, proba)
    is_spam = label == "Spam"
    color = SPAM_COLOR if is_spam else HAM_COLOR
    icon_bg = "#FDECEC" if is_spam else "#E7F0FB"
    rows = ""
    for k in MODELS:
        p = result["probas"][k]
        k_color = SPAM_COLOR if verdict(k, p) == "Spam" else HAM_COLOR
        rows += (
            f'<span style="font-weight:500">{MODELS[k]["short"]}</span>'
            f'<span class="mono">{fmt_proba(p)}</span>'
            f'<span style="color:{k_color};font-weight:600">{verdict(k, p)}</span>'
        )
    st.markdown(
        f"""
        <div class="verdict">
          <div class="verdict-icon" style="background:{icon_bg}">
            {ICON_SPAM if is_spam else ICON_HAM}</div>
          <div><div class="verdict-label" style="color:{color}">{label}</div>
            <div class="muted">{MODELS[model_key]["by"]}</div></div>
        </div>
        {proba_bar(model_key, proba)}
        <div class="section-sep"></div>
        <div style="font-weight:600;font-size:0.9rem">Both models on this message</div>
        <div class="compare">
          <span class="muted">Model</span><span class="muted">P(spam)</span>
          <span class="muted">Verdict</span>{rows}
        </div>
        <div class="section-sep"></div>
        <div style="font-weight:600;font-size:0.9rem;margin-bottom:4px">
          Words driving the decision</div>
        <div class="words">{highlighted_words(result["effects"][model_key])}</div>
        <div class="muted">
          <span class="legend-dot" style="background:rgba(180,35,24,.45);margin-left:0"></span>
          pushes towards spam
          <span class="legend-dot" style="background:rgba(29,78,137,.45)"></span>
          pushes towards legit · each word is removed in turn to measure its effect.
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_analyse:
    col_input, col_result = st.columns([3, 2], gap="medium")

    with col_input, st.container(border=True):
        st.markdown("#### Message to analyze")
        st.segmented_control(
            "Model",
            options=list(MODELS),
            format_func=lambda k: MODELS[k]["label"],
            default="network",
            key="model_choice",
        )
        user_input = st.text_area(
            "SMS in English (both models were trained on English messages)",
            height=140,
            key="user_input_area",
        )
        st.pills(
            "Examples",
            options=[*EXAMPLES, RANDOM_EXAMPLE],
            key="example_pick",
            on_change=use_example,
        )
        predict_button = st.button(
            "Analyze message", type="primary", icon=":material/search:", key="predict"
        )

    if predict_button:
        message = user_input.strip()
        if len(message) < MIN_LENGTH:
            st.session_state.last_result = {"error": True}
        else:
            with st.spinner("Analyzing…"):
                probas = {k: spam_probabilities(k, [message])[0] for k in MODELS}
                effects = {k: word_effects(k, message) for k in MODELS}
            model_key = selected_model()
            st.session_state.history.append(
                {
                    "Message": message,
                    "Model": MODELS[model_key]["short"],
                    "Verdict": verdict(model_key, probas[model_key]),
                    **{MODELS[k]["short"]: p for k, p in probas.items()},
                }
            )
            st.session_state.last_result = {
                "error": False,
                "message": message,
                "probas": probas,
                "effects": effects,
            }

    with col_result, st.container(border=True):
        st.markdown("#### Verdict")
        result = st.session_state.last_result
        if result is None:
            st.markdown(
                '<div class="placeholder">Type or pick an SMS, then run the analysis '
                "to see the verdict here.</div>",
                unsafe_allow_html=True,
            )
        elif result["error"]:
            st.error(
                f"Message too short: enter at least {MIN_LENGTH} characters.",
                icon=":material/error:",
            )
        else:
            render_result(result)

    if st.session_state.history:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown("#### Session history")
                st.space("stretch")
                history = pd.DataFrame(st.session_state.history)
                st.download_button(
                    "Export CSV",
                    data=history.to_csv(index=False).encode("utf-8"),
                    file_name="sms_history.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.button("Clear", icon=":material/delete:", on_click=clear_history)
            st.dataframe(
                history.iloc[::-1],
                hide_index=True,
                width="stretch",
                column_config={
                    "Message": st.column_config.TextColumn(width="large"),
                    "Model": st.column_config.TextColumn(width="small"),
                    "Verdict": st.column_config.TextColumn(width="small"),
                    **{
                        MODELS[k]["short"]: st.column_config.ProgressColumn(
                            f"P(spam) · {MODELS[k]['short'].lower()}",
                            min_value=0.0,
                            max_value=1.0,
                            format="percent",
                        )
                        for k in MODELS
                    },
                },
            )


# ── Performance ───────────────────────────────────────────────────────────────
def confusion_figure(cm):
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    text = [
        [
            plural(cm["tn"], "legit recognized", "legit recognized").replace(" ", "<br>", 1),
            plural(cm["fp"], "false alarm", "false alarms").replace(" ", "<br>", 1),
        ],
        [
            plural(cm["fn"], "spam missed", "spams missed").replace(" ", "<br>", 1),
            plural(cm["tp"], "spam caught", "spams caught").replace(" ", "<br>", 1),
        ],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=[[math.log1p(v) for v in row] for row in z],
            x=["Predicted legit", "Predicted spam"],
            y=["Actual legit", "Actual spam"],
            text=text,
            texttemplate="%{text}",
            hoverinfo="skip",
            colorscale=[[0, "#F3F4F6"], [1, ACCENT]],
            showscale=False,
            xgap=4,
            ygap=4,
        )
    )
    fig.update_traces(textfont={"size": 14})
    fig.update_layout(
        height=280,
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        yaxis={"autorange": "reversed"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#111827"},
    )
    return fig


with tab_perf:
    st.markdown("#### Comparison on the test set")
    st.caption(
        f"Same split for every model: {fmt_int(metrics['data']['test'])} unique messages never "
        f"seen in training. Each threshold targets "
        f"{fmt_pct(metrics['min_precision_target'], 0)} precision on out-of-fold train predictions."
    )
    compared = [
        ("Word + character network", metrics["network"]),
        ("TF-IDF + logistic regression baseline", metrics["baseline"]),
    ]
    if "original_network" in metrics:
        compared.append(("First version of the network", metrics["original_network"]))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model": name,
                    "Precision": fmt_pct(m["test"]["precision"]),
                    "Recall": fmt_pct(m["test"]["recall"]),
                    "Spam F1": fmt_num(m["test"]["f1"]),
                    "False alarms": m["test"]["confusion_matrix"]["fp"],
                    "Spams missed": m["test"]["confusion_matrix"]["fn"],
                    "Threshold": fmt_num(m["threshold"]),
                }
                for name, m in compared
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={"Model": st.column_config.TextColumn(width="large")},
    )
    for col, key in zip(st.columns(2, gap="medium"), MODELS, strict=True):
        with col.container(border=True):
            st.markdown(f"**{MODELS[key]['label']}** · F1 {fmt_num(metrics[key]['test']['f1'])}")
            st.plotly_chart(
                confusion_figure(metrics[key]["test"]["confusion_matrix"]),
                width="stretch",
                config={"displayModeBar": False},
                key=f"cm_{key}",
            )
    st.caption(
        "Reading: on this test set the network is the most cautious (a single false alarm) "
        "while the baseline catches a few more spams. The F1 gap comes down to a few messages."
    )


# ── Method ─────────────────────────────────────────────────────────────────────
with tab_method:
    data = metrics["data"]
    col_data, col_limits = st.columns([3, 2], gap="medium")
    with col_data.container(border=True):
        st.markdown("#### Data and protocol")
        st.markdown(
            f"""
- **Source**: UCI SMS Spam Collection, English SMS from the 2010s.
- **Duplicates removed**: {fmt_int(data["messages_raw"])} → {fmt_int(data["messages_unique"])}
  unique messages, {fmt_pct(data["spam_share"])} of them spam.
- **Split**: stratified 80 / 20 (seed 42), i.e. {fmt_int(data["train"])} training
  and {fmt_int(data["test"])} test messages.
- **Threshold**: chosen on out-of-fold train predictions, never on the test set.
- **Network architecture**: selected by 5-fold cross-validation on the train set.
"""
        )
    with col_limits.container(border=True):
        st.markdown("#### Limitations")
        st.markdown(
            "The messages are English SMS from the 2010s: today's scams "
            "(short links, fake parcels, crypto) are barely represented. "
            "This is an educational project, not a production filter."
        )
        st.markdown(
            f"[Code and notebook on GitHub]({REPO_URL}) · "
            f"[Retrain the models]({REPO_URL}/blob/main/train.py)"
        )
    with st.container(border=True):
        st.markdown("#### The two models")
        net = metrics["network"]
        st.dataframe(
            pd.DataFrame(
                {
                    "Step": ["Features", "Classifier", "Threshold", "Size"],
                    MODELS["network"]["label"]: [
                        MODELS["network"]["features"],
                        MODELS["network"]["classifier"],
                        fmt_num(net["threshold"]),
                        f"{fmt_int(net['parameters'])} parameters · "
                        f"{fmt_num(net['size_kb'] / 1024, 1)} MB",
                    ],
                    MODELS["baseline"]["label"]: [
                        MODELS["baseline"]["features"],
                        MODELS["baseline"]["classifier"],
                        fmt_num(metrics["baseline"]["threshold"]),
                        f"{fmt_num(metrics['baseline']['size_kb'] / 1024, 1)} MB",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "The network reads raw text: word and character vectorization live inside the "
            "Keras model, so the app has no preprocessing to reproduce."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;
                color:{MUTED};font-size:0.85rem;margin-top:1.5rem;padding-top:1rem;
                border-top:1px solid #E5E7EB">
      <span>Data: UCI SMS Spam Collection · {fmt_int(metrics["data"]["messages_unique"])}
        unique messages · educational project.</span>
      <span>Christophe Noret · <a href="{REPO_URL}" target="_blank">GitHub</a> ·
        <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a></span>
    </div>
    """,
    unsafe_allow_html=True,
)
