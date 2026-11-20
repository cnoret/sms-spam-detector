"""Training protocol: data loading, threshold choice, metrics and reproducibility."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import train


@pytest.fixture(scope="module")
def data():
    return train.load_data()


def test_load_data_removes_duplicates(data):
    assert len(data) == 5158
    assert data.message.is_unique
    assert set(data.label.unique()) == {0, 1}


def test_load_data_rebuilds_messages_split_over_columns(tmp_path):
    csv = tmp_path / "spam.csv"
    pd.DataFrame(
        {
            "v1": ["ham", "spam", "ham"],
            "v2": ["see you", "WIN cash", "see you"],
            "Unnamed: 2": [None, "now", None],
        }
    ).to_csv(csv, index=False, encoding="latin-1")
    df = train.load_data(csv)
    assert df.message.tolist() == ["see you", "WIN cash now"]
    assert df.label.tolist() == [0, 1]


def test_choose_threshold_meets_precision_target():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # 0.6 is the lowest threshold with no false positive.
    assert train.choose_threshold(y, proba, min_precision=1.0) == pytest.approx(0.6)


def test_choose_threshold_falls_back_to_half_when_target_unreachable():
    y = np.array([1, 0])
    proba = np.array([0.2, 0.9])
    assert train.choose_threshold(y, proba, min_precision=1.0) == 0.5


def test_scores():
    s = train.scores([0, 0, 1, 1], [0, 1, 1, 1])
    assert s["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert s["precision"] == pytest.approx(0.667)
    assert s["recall"] == 1.0


def test_train_baseline_reproduces_committed_metrics(data, tmp_path, monkeypatch):
    """Retraining from scratch gives the numbers shown in the README and the app."""
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "model.joblib")
    tr, te = train_test_split(
        data, test_size=0.2, stratify=data.label, random_state=train.SEED
    )
    result = train.train_baseline(tr, te)

    committed = json.loads(train.METRICS_PATH.read_text(encoding="utf-8"))
    assert committed["data"]["train"] == len(tr)
    assert committed["data"]["test"] == len(te)
    assert result["threshold"] == pytest.approx(committed["deployed"]["threshold"], abs=0.01)
    assert result["test"]["f1"] == pytest.approx(committed["deployed"]["test"]["f1"], abs=0.005)
