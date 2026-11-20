"""The committed model artifacts behave as their metrics say."""

import json

import joblib
import pytest

import train

MODEL_KEYS = ("network", "baseline")


@pytest.fixture(scope="module")
def metrics():
    return json.loads(train.METRICS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(train.MODEL_PATH)


@pytest.fixture(scope="module")
def network():
    import keras

    return keras.models.load_model(train.NETWORK_PATH)


@pytest.fixture(scope="module")
def spam_proba(bundle, network):
    def proba(model_key, message):
        if model_key == "network":
            return float(train.network_proba(network, [message])[0])
        return float(bundle["pipeline"].predict_proba([message])[0, 1])

    return proba


def test_baseline_threshold_matches_metrics(bundle, metrics):
    assert bundle["threshold"] == pytest.approx(metrics["baseline"]["threshold"], abs=1e-3)


@pytest.mark.parametrize("model_key", [*MODEL_KEYS, "original_network"])
def test_metrics_are_consistent(metrics, model_key):
    test = metrics[model_key]["test"]
    cm = test["confusion_matrix"]
    assert sum(cm.values()) == metrics["data"]["test"]
    assert test["precision"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fp"]), abs=1e-3)
    assert test["recall"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fn"]), abs=1e-3)


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_deployed_models_keep_precision_target(metrics, model_key):
    assert metrics[model_key]["test"]["precision"] >= metrics["min_precision_target"] - 0.02


def test_network_improves_on_the_original(metrics):
    assert metrics["network"]["test"]["f1"] > metrics["original_network"]["test"]["f1"] + 0.05


def test_network_takes_raw_text_with_line_breaks(network):
    probas = train.network_proba(network, ["Call me back\r\nplease", "", "£££ 0800 !!!"])
    assert probas.shape == (3,)
    assert ((probas >= 0) & (probas <= 1)).all()


@pytest.mark.parametrize("model_key", MODEL_KEYS)
@pytest.mark.parametrize(
    "message",
    [
        "URGENT! You have won a £1000 cash prize. Call 09061701461 to claim now.",
        "FREE entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121",
    ],
)
def test_flags_classic_spam(spam_proba, metrics, model_key, message):
    assert spam_proba(model_key, message) >= metrics[model_key]["threshold"]


@pytest.mark.parametrize("model_key", MODEL_KEYS)
@pytest.mark.parametrize(
    "message",
    [
        "Hey, are we still on for dinner tonight?",
        "I'll be home in 10 minutes, can you put the kettle on",
    ],
)
def test_keeps_everyday_messages(spam_proba, metrics, model_key, message):
    assert spam_proba(model_key, message) < metrics[model_key]["threshold"]
