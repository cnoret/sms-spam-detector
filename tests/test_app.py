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


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


@pytest.fixture
def app():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def page_html(at):
    return " ".join(m.value for m in at.markdown)


def predict(at, message):
    at.text_area(key="user_input_area").set_value(message)
    at.button(key="predict").click()
    at.run()
    assert not at.exception
    return page_html(at)


def test_app_starts_with_an_example_and_the_network(app):
    assert "SMS Spam Detector" in page_html(app)
    assert app.text_area(key="user_input_area").value
    assert app.segmented_control(key="model_choice").value == "network"
    assert [t.label.split()[-1] for t in app.tabs] == ["Analyze", "Performance", "Method"]


def test_header_shows_the_network_scores(app):
    values = [m.value for m in app.metric]
    assert fmt(METRICS["network"]["test"]["f1"]) in values


def test_performance_tab_compares_both_models(app):
    perf = app.tabs[1]
    table = perf.dataframe[0].value
    for key in ("network", "baseline"):
        assert fmt(METRICS[key]["test"]["f1"]) in table["Spam F1"].tolist()
    # One confusion matrix per model.
    assert len(perf.get("plotly_chart")) == 2


def test_predicts_spam(app):
    html = predict(app, SPAM)
    assert ">Spam<" in html
    assert len(app.session_state.history) == 1
    # Both models' scores, side by side.
    assert html.count('class="mono"') >= 3


def test_predicts_non_spam(app):
    html = predict(app, HAM)
    assert ">Legit<" in html


@pytest.mark.parametrize("model_key", ["network", "baseline"])
def test_result_follows_the_selected_model(app, model_key):
    predict(app, SPAM)
    app.segmented_control(key="model_choice").set_value(model_key).run()
    assert not app.exception
    proba = app.session_state.last_result["probas"][model_key]
    assert f"width:{proba * 100:.1f}%" in page_html(app)


def test_explains_the_decision_word_by_word(app):
    html = predict(app, SPAM)
    effects = app.session_state.last_result["effects"]
    assert [w for w, _ in effects["network"]] == SPAM.split()
    # The strongest spam signal is highlighted.
    top_word = max(effects["network"], key=lambda we: we[1])[0]
    assert f">{top_word}</mark>" in html.replace("&#x27;", "'")


def test_explanation_escapes_html(app):
    html = predict(app, "<b>FREE</b> entry, text WIN to 80086 now")
    assert "<b>FREE</b>" not in html
    assert "&lt;b&gt;FREE&lt;/b&gt;" in html


def test_example_pills_fill_the_text_area(app):
    app.pills(key="example_pick").set_value("Personal").run()
    assert not app.exception
    assert app.text_area(key="user_input_area").value == HAM
    assert app.pills(key="example_pick").value is None


def test_history_records_the_model_and_exports_csv(app):
    app.segmented_control(key="model_choice").set_value("baseline").run()
    predict(app, HAM)
    entry = app.session_state.history[0]
    assert entry["Model"] == "Baseline"
    assert entry["Verdict"] == "Legit"
    assert len(app.get("download_button")) == 1


def test_clear_empties_the_history(app):
    predict(app, HAM)
    next(b for b in app.button if b.label == "Clear").click().run()
    assert app.session_state.history == []
    assert app.session_state.last_result is None


def test_rejects_too_short_message(app):
    predict(app, "hi")
    assert "too short" in app.error[0].value
    assert app.session_state.history == []
