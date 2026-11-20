"""End-to-end checks of the Streamlit app with AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app():
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def predict(at, message):
    at.text_area(key="user_input_area").set_value(message)
    next(b for b in at.button if "Predict" in b.label).click()
    at.run()
    assert not at.exception
    return " ".join(m.value for m in at.markdown)


def test_app_starts_with_an_example(app):
    assert app.title[0].value == "📲 SMS Spam Detector"
    assert app.text_area(key="user_input_area").value
    assert len(app.metric) == 4


def test_predicts_spam(app):
    html = predict(app, "WINNER!! You have won a £900 prize. Call 09061701461 to claim now")
    assert ">Spam<" in html
    assert len(app.session_state.history) == 1


def test_predicts_non_spam(app):
    html = predict(app, "Are we still meeting for lunch at noon tomorrow?")
    assert ">Non-Spam<" in html


def test_rejects_too_short_message(app):
    predict(app, "hi")
    assert "too short" in app.error[0].value
    assert app.session_state.history == []
