"""End-to-end checks of the Streamlit app with AppTest."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
METRICS = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))

SPAM = "WINNER!! You have won a £900 prize. Call 09061701461 to claim now"
HAM = "Are we still meeting for lunch at noon tomorrow?"


@pytest.fixture
def app():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def predict(at, message):
    at.text_area(key="user_input_area").set_value(message)
    next(b for b in at.button if "Predict" in b.label).click()
    at.run()
    assert not at.exception
    return " ".join(m.value for m in at.markdown)


def test_app_starts_with_an_example_and_the_network(app):
    assert app.title[0].value == "📲 SMS Spam Detector"
    assert app.text_area(key="user_input_area").value
    assert app.radio(key="model_choice").value == "network"


def test_sidebar_compares_both_models(app):
    sidebar = " ".join(m.value for m in app.sidebar.markdown)
    for key in ("network", "baseline"):
        assert f"{METRICS[key]['test']['f1']:.3f}" in sidebar


def test_predicts_spam(app):
    html = predict(app, SPAM)
    assert ">Spam<" in html
    assert len(app.session_state.history) == 1
    # Both models' scores, side by side.
    assert len(app.metric) == 2
    assert all("Spam" in m.label for m in app.metric)


def test_predicts_non_spam(app):
    html = predict(app, HAM)
    assert ">Non-Spam<" in html


@pytest.mark.parametrize("model_key", ["network", "baseline"])
def test_result_follows_the_selected_model(app, model_key):
    predict(app, SPAM)
    app.radio(key="model_choice").set_value(model_key).run()
    assert not app.exception
    proba = app.session_state.last_result["probas"][model_key]
    html = " ".join(m.value for m in app.markdown)
    assert f"<b>{proba * 100:.1f}%</b>" in html


def test_history_records_the_model(app):
    app.radio(key="model_choice").set_value("baseline").run()
    predict(app, HAM)
    entry = app.session_state.history[0]
    assert entry["Model"] == "📏 Baseline"
    assert entry["Label"] == "Non-Spam"


def test_rejects_too_short_message(app):
    predict(app, "hi")
    assert "too short" in app.error[0].value
    assert app.session_state.history == []
