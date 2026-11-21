"""Streamlit app: detect spam SMS with a neural network, compared with a baseline.

Both models are trained and evaluated by train.py under the same protocol:
- the neural network (word embeddings + character CNN, Keras) is the model studied,
- TF-IDF + logistic regression is the comparison baseline.

The interface is in French; the models read English SMS.
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
    page_title="Détecteur de spam SMS",
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
        "label": "Réseau de neurones",
        "short": "Réseau",
        "by": "selon le réseau de neurones",
        "features": "Embeddings de mots + CNN sur les caractères",
        "classifier": "Keras · couche dense + sigmoïde",
    },
    "baseline": {
        "label": "Baseline TF-IDF",
        "short": "Baseline",
        "by": "selon la baseline TF-IDF",
        "features": "TF-IDF · n-grammes de caractères (2 à 5)",
        "classifier": "Régression logistique (classes équilibrées)",
    },
}

EXAMPLES = {
    "Gain fictif": "WINNER!! You have won a £900 prize. Call 09061701461 to claim now",
    "Phishing bancaire": (
        "Alert! Your bank account has suspicious activity. "
        "Click to verify: http://bank.fakealert.com"
    ),
    "Message perso": "Are we still meeting for lunch at noon tomorrow?",
}

RANDOM_EXAMPLE = ":material/shuffle: Au hasard"
MIN_LENGTH = 5
MAX_EXPLAINED_WORDS = 60
REPO_URL = "https://github.com/cnoret/sms-spam-detector"
LINKEDIN_URL = "https://www.linkedin.com/in/christophenoret/"


# ── Models ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du réseau de neurones…")
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
    return "Spam" if proba >= metrics[model_key]["threshold"] else "Légitime"


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


# ── Formatting (French conventions) ──────────────────────────────────────────
NBSP = "\u00a0"


def fr_num(x, digits=3):
    return f"{x:.{digits}f}".replace(".", ",")


def fr_pct(x, digits=1):
    return f"{x * 100:.{digits}f}{NBSP}%".replace(".", ",")


def fr_int(n):
    return f"{n:,}".replace(",", NBSP)


def fr_proba(p):
    """Probability as a percentage, without claiming 0 % or 100 %."""
    if p >= 0.999:
        return f">{NBSP}99,9{NBSP}%"
    if p < 0.001:
        return f"<{NBSP}0,1{NBSP}%"
    return fr_pct(p)


def plural(n, singular, plural_form):
    return f"{fr_int(n)} {singular if n <= 1 else plural_form}"


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
      <h1 class="app-title" style="margin:0;padding:0;font-size:1.45rem">Détecteur de spam SMS</h1>
      <span class="app-badge">Réseau Keras · mots + caractères</span>
    </div>
    <p class="app-subtitle">Un réseau de neurones entraîné sur la UCI SMS Spam Collection,
    comparé à une baseline TF-IDF sur le même jeu de test.</p>
    """,
    unsafe_allow_html=True,
)

kpis = [
    (
        "F1 spam · réseau",
        fr_num(network_test["f1"]),
        f"baseline {fr_num(baseline_test['f1'])}"
        + (
            f" · v1 {fr_num(metrics['original_network']['test']['f1'])}"
            if "original_network" in metrics
            else ""
        ),
    ),
    (
        "Précision spam",
        fr_pct(network_test["precision"]),
        plural(
            network_test["confusion_matrix"]["fp"],
            "SMS légitime signalé à tort",
            "SMS légitimes signalés à tort",
        ),
    ),
    (
        "Rappel spam",
        fr_pct(network_test["recall"]),
        f"{network_test['confusion_matrix']['fn']} spams manqués sur "
        f"{network_test['confusion_matrix']['fn'] + network_test['confusion_matrix']['tp']}",
    ),
    ("Jeu de test", fr_int(metrics["data"]["test"]), "messages uniques, jamais vus"),
]
for col, (label, value, note) in zip(st.columns(4), kpis, strict=True):
    with col.container(border=True):
        st.metric(label, value)
        st.caption(note)

tab_analyse, tab_perf, tab_method = st.tabs(
    [":material/search: Analyser", ":material/monitoring: Performance", ":material/school: Méthode"]
)


# ── Analyser ──────────────────────────────────────────────────────────────────
def proba_bar(model_key, proba):
    threshold = metrics[model_key]["threshold"]
    color = SPAM_COLOR if proba >= threshold else HAM_COLOR
    return f"""
      <div class="proba-row"><span class="muted">Probabilité de spam</span>
        <span class="mono">{fr_proba(proba)}</span></div>
      <div class="proba-track">
        <div class="proba-fill" style="width:{proba * 100:.1f}%;background:{color}"></div>
        <div class="proba-threshold" style="left:{threshold * 100:.1f}%"></div>
      </div>
      <div class="muted" style="margin-top:6px">Trait noir : seuil de décision
        {fr_num(threshold)}, fixé pour viser {fr_pct(metrics["min_precision_target"], 0)}
        de précision.</div>
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
            f'<span class="mono">{fr_proba(p)}</span>'
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
        <div style="font-weight:600;font-size:0.9rem">Les deux modèles sur ce message</div>
        <div class="compare">
          <span class="muted">Modèle</span><span class="muted">P(spam)</span>
          <span class="muted">Verdict</span>{rows}
        </div>
        <div class="section-sep"></div>
        <div style="font-weight:600;font-size:0.9rem;margin-bottom:4px">
          Mots qui pèsent dans la décision</div>
        <div class="words">{highlighted_words(result["effects"][model_key])}</div>
        <div class="muted">
          <span class="legend-dot" style="background:rgba(180,35,24,.45);margin-left:0"></span>
          pousse vers spam
          <span class="legend-dot" style="background:rgba(29,78,137,.45)"></span>
          pousse vers légitime · on retire chaque mot et on mesure l'écart de score.
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_analyse:
    col_input, col_result = st.columns([3, 2], gap="medium")

    with col_input, st.container(border=True):
        st.markdown("#### Message à analyser")
        st.segmented_control(
            "Modèle",
            options=list(MODELS),
            format_func=lambda k: MODELS[k]["label"],
            default="network",
            key="model_choice",
        )
        user_input = st.text_area(
            "SMS en anglais (les modèles ont été entraînés sur des SMS anglais)",
            height=140,
            key="user_input_area",
        )
        st.pills(
            "Exemples",
            options=[*EXAMPLES, RANDOM_EXAMPLE],
            key="example_pick",
            on_change=use_example,
        )
        predict_button = st.button(
            "Analyser le message", type="primary", icon=":material/search:", key="predict"
        )

    if predict_button:
        message = user_input.strip()
        if len(message) < MIN_LENGTH:
            st.session_state.last_result = {"error": True}
        else:
            with st.spinner("Analyse en cours…"):
                probas = {k: spam_probabilities(k, [message])[0] for k in MODELS}
                effects = {k: word_effects(k, message) for k in MODELS}
            model_key = selected_model()
            st.session_state.history.append(
                {
                    "Message": message,
                    "Modèle": MODELS[model_key]["short"],
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
                '<div class="placeholder">Saisissez ou choisissez un SMS, puis lancez '
                "l'analyse pour voir le verdict ici.</div>",
                unsafe_allow_html=True,
            )
        elif result["error"]:
            st.error(
                f"Message trop court : saisissez au moins {MIN_LENGTH} caractères.",
                icon=":material/error:",
            )
        else:
            render_result(result)

    if st.session_state.history:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown("#### Historique de la session")
                st.space("stretch")
                history = pd.DataFrame(st.session_state.history)
                st.download_button(
                    "Exporter en CSV",
                    data=history.to_csv(index=False).encode("utf-8"),
                    file_name="historique_sms.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.button("Effacer", icon=":material/delete:", on_click=clear_history)
            st.dataframe(
                history.iloc[::-1],
                hide_index=True,
                width="stretch",
                column_config={
                    "Message": st.column_config.TextColumn(width="large"),
                    "Modèle": st.column_config.TextColumn(width="small"),
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
            plural(cm["tn"], "légitime reconnu", "légitimes reconnus").replace(" ", "<br>", 1),
            plural(cm["fp"], "fausse alerte", "fausses alertes").replace(" ", "<br>", 1),
        ],
        [
            plural(cm["fn"], "spam manqué", "spams manqués").replace(" ", "<br>", 1),
            plural(cm["tp"], "spam détecté", "spams détectés").replace(" ", "<br>", 1),
        ],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=[[math.log1p(v) for v in row] for row in z],
            x=["Prédit légitime", "Prédit spam"],
            y=["Vrai légitime", "Vrai spam"],
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
    st.markdown("#### Comparaison sur le jeu de test")
    st.caption(
        f"Même découpage pour tous : {fr_int(metrics['data']['test'])} messages uniques jamais "
        f"vus à l'entraînement. Seuil fixé pour viser une précision de "
        f"{fr_pct(metrics['min_precision_target'], 0)} sur les prédictions hors pli du train."
    )
    compared = [
        ("Réseau mots + caractères", metrics["network"]),
        ("Baseline TF-IDF + régression logistique", metrics["baseline"]),
    ]
    if "original_network" in metrics:
        compared.append(("Première version du réseau", metrics["original_network"]))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Modèle": name,
                    "Précision": fr_pct(m["test"]["precision"]),
                    "Rappel": fr_pct(m["test"]["recall"]),
                    "F1 spam": fr_num(m["test"]["f1"]),
                    "Fausses alertes": m["test"]["confusion_matrix"]["fp"],
                    "Spams manqués": m["test"]["confusion_matrix"]["fn"],
                    "Seuil": fr_num(m["threshold"]),
                }
                for name, m in compared
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={"Modèle": st.column_config.TextColumn(width="large")},
    )
    for col, key in zip(st.columns(2, gap="medium"), MODELS, strict=True):
        with col.container(border=True):
            st.markdown(f"**{MODELS[key]['label']}** · F1 {fr_num(metrics[key]['test']['f1'])}")
            st.plotly_chart(
                confusion_figure(metrics[key]["test"]["confusion_matrix"]),
                width="stretch",
                config={"displayModeBar": False},
                key=f"cm_{key}",
            )
    st.caption(
        "Lecture : sur ce jeu de test, le réseau est le plus prudent (une seule fausse alerte) "
        "et la baseline rattrape un peu plus de spams. L'écart de F1 tient à quelques messages."
    )


# ── Méthode ───────────────────────────────────────────────────────────────────
with tab_method:
    data = metrics["data"]
    col_data, col_limits = st.columns([3, 2], gap="medium")
    with col_data.container(border=True):
        st.markdown("#### Données et protocole")
        st.markdown(
            f"""
- **Source** : UCI SMS Spam Collection, SMS en anglais des années 2010.
- **Doublons retirés** : {fr_int(data["messages_raw"])} → {fr_int(data["messages_unique"])}
  messages uniques, dont {fr_pct(data["spam_share"])} de spams.
- **Découpage** : 80 / 20 stratifié (graine 42), soit {fr_int(data["train"])} messages
  d'entraînement et {fr_int(data["test"])} de test.
- **Seuil** : choisi sur les prédictions hors pli du train, jamais sur le test.
- **Architecture du réseau** : retenue en validation croisée à 5 plis sur le train.
"""
        )
    with col_limits.container(border=True):
        st.markdown("#### Limites")
        st.markdown(
            "Les SMS datent des années 2010 et sont en anglais : les arnaques actuelles "
            "(liens raccourcis, faux colis, crypto) y sont peu représentées. "
            "C'est un projet pédagogique, pas un filtre de production."
        )
        st.markdown(
            f"[Code et notebook sur GitHub]({REPO_URL}) · "
            f"[Ré-entraîner les modèles]({REPO_URL}/blob/main/train.py)"
        )
    with st.container(border=True):
        st.markdown("#### Les deux modèles")
        net = metrics["network"]
        st.dataframe(
            pd.DataFrame(
                {
                    "Étape": ["Représentation", "Classifieur", "Seuil", "Taille"],
                    MODELS["network"]["label"]: [
                        MODELS["network"]["features"],
                        MODELS["network"]["classifier"],
                        fr_num(net["threshold"]),
                        f"{fr_int(net['parameters'])} paramètres · "
                        f"{fr_num(net['size_kb'] / 1024, 1)} Mo",
                    ],
                    MODELS["baseline"]["label"]: [
                        MODELS["baseline"]["features"],
                        MODELS["baseline"]["classifier"],
                        fr_num(metrics["baseline"]["threshold"]),
                        f"{fr_num(metrics['baseline']['size_kb'] / 1024, 1)} Mo",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Le réseau lit le texte brut : la vectorisation (mots et caractères) est intégrée "
            "au modèle Keras, il n'y a donc aucun prétraitement à reproduire côté app."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;
                color:{MUTED};font-size:0.85rem;margin-top:1.5rem;padding-top:1rem;
                border-top:1px solid #E5E7EB">
      <span>Données : UCI SMS Spam Collection · {fr_int(metrics["data"]["messages_unique"])}
        messages uniques · projet pédagogique.</span>
      <span>Christophe Noret · <a href="{REPO_URL}" target="_blank">GitHub</a> ·
        <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a></span>
    </div>
    """,
    unsafe_allow_html=True,
)
