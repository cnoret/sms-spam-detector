"""The committed model artifact behaves as its metrics say."""

import json

import joblib
import pytest

import train


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(train.MODEL_PATH)


@pytest.fixture(scope="module")
def metrics():
    return json.loads(train.METRICS_PATH.read_text(encoding="utf-8"))


def test_threshold_matches_metrics(bundle, metrics):
    assert bundle["threshold"] == pytest.approx(metrics["deployed"]["threshold"], abs=1e-3)


def test_metrics_are_consistent(metrics):
    test = metrics["deployed"]["test"]
    cm = test["confusion_matrix"]
    assert sum(cm.values()) == metrics["data"]["test"]
    assert test["precision"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fp"]), abs=1e-3)
    assert test["recall"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fn"]), abs=1e-3)
    assert test["precision"] >= metrics["min_precision_target"] - 0.02


@pytest.mark.parametrize(
    "message",
    [
        "URGENT! You have won a £1000 cash prize. Call 09061701461 to claim now.",
        "FREE entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121",
    ],
)
def test_flags_classic_spam(bundle, message):
    proba = bundle["pipeline"].predict_proba([message])[0, 1]
    assert proba >= bundle["threshold"]


@pytest.mark.parametrize(
    "message",
    [
        "Hey, are we still on for dinner tonight?",
        "I'll be home in 10 minutes, can you put the kettle on",
    ],
)
def test_keeps_everyday_messages(bundle, message):
    proba = bundle["pipeline"].predict_proba([message])[0, 1]
    assert proba < bundle["threshold"]
