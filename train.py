"""Train the SMS spam classifier and write its evaluation to models/metrics.json.

Protocol (fixes the leaks found in the original notebook):
- messages are deduplicated before splitting (414 duplicate rows in the raw file),
- one stratified 80/20 train/test split, the test set is used once at the end,
- the decision threshold is chosen on out-of-fold predictions of the train set,
- everything the model learns (vocabulary, weights) comes from the train set only.

Two models are compared under the same protocol:
- the deployed model: TF-IDF on character n-grams + logistic regression,
- the original neural network (Embedding + Dense), retrained for reference.

Usage:
    python train.py                 # baseline + network comparison
    python train.py --skip-network  # baseline only (no TensorFlow needed)
"""

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
DATA_PATH = Path("data/spam.csv")
MODEL_PATH = Path("models/spam_classifier.joblib")
METRICS_PATH = Path("models/metrics.json")

# Blocking a legitimate SMS costs more than letting a spam through:
# keep the highest recall that still meets this precision on validation.
MIN_PRECISION = 0.98


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the UCI file, rebuild messages split over extra columns, deduplicate."""
    raw = pd.read_csv(path, encoding="latin-1")
    text_cols = [c for c in raw.columns if c != "v1"]
    df = pd.DataFrame(
        {
            "message": raw[text_cols].fillna("").agg(" ".join, axis=1).str.strip(),
            "label": raw["v1"].map({"ham": 0, "spam": 1}),
        }
    )
    return df.drop_duplicates("message").reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2
                ),
            ),
            (
                "clf",
                LogisticRegression(C=3, class_weight="balanced", max_iter=5000),
            ),
        ]
    )


def choose_threshold(y_true, proba, min_precision: float = MIN_PRECISION) -> float:
    """Lowest threshold (so highest recall) whose precision reaches min_precision."""
    precision, _, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(precision[:-1] >= min_precision)[0]
    return float(thresholds[ok[0]]) if len(ok) else 0.5


def scores(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred), 3),
        "recall": round(recall_score(y_true, y_pred), 3),
        "f1": round(f1_score(y_true, y_pred), 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(
        pipeline, train.message, train.label, cv=cv, method="predict_proba"
    )[:, 1]
    threshold = choose_threshold(train.label, oof)

    pipeline.fit(train.message, train.label)
    proba = pipeline.predict_proba(test.message)[:, 1]
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold}, MODEL_PATH)

    return {
        "name": "TF-IDF (character n-grams) + logistic regression",
        "threshold": round(threshold, 3),
        "test": scores(test.label, (proba >= threshold).astype(int)),
        "test_at_0.5": scores(test.label, (proba >= 0.5).astype(int)),
        "size_kb": round(MODEL_PATH.stat().st_size / 1024),
    }


def train_network(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Original Embedding + Dense network, retrained under the same protocol."""
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.text import Tokenizer

    from preprocessing import MAXLEN, NUM_WORDS, OOV_TOKEN, clean_text, to_padded_sequences

    tf.keras.utils.set_random_seed(SEED)
    fit_df, val_df = train_test_split(
        train, test_size=0.2, stratify=train.label, random_state=SEED
    )
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts([clean_text(t) for t in fit_df.message])

    def encode(df):
        return to_padded_sequences(list(df.message), tokenizer, maxlen=MAXLEN)

    x_fit, x_val, x_test = encode(fit_df), encode(val_df), encode(test)
    counts = fit_df.label.value_counts()
    class_weight = {k: len(fit_df) / (2 * v) for k, v in counts.items()}

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(MAXLEN,)),
            layers.Embedding(NUM_WORDS + 2, 128),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(
        x_fit,
        fit_df.label.values,
        validation_data=(x_val, val_df.label.values),
        epochs=30,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=0,
    )
    threshold = choose_threshold(val_df.label, model.predict(x_val, verbose=0).ravel())
    proba = model.predict(x_test, verbose=0).ravel()
    return {
        "name": "Word embedding + Dense network (original architecture)",
        "threshold": round(threshold, 3),
        "test": scores(test.label, (proba >= threshold).astype(int)),
        "test_at_0.5": scores(test.label, (proba >= 0.5).astype(int)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-network", action="store_true")
    args = parser.parse_args()

    df = load_data()
    train, test = train_test_split(
        df, test_size=0.2, stratify=df.label, random_state=SEED
    )
    metrics = {
        "trained_on": date.today().isoformat(),
        "scikit_learn": sklearn.__version__,
        "data": {
            "messages_raw": int(pd.read_csv(DATA_PATH, encoding="latin-1").shape[0]),
            "messages_unique": len(df),
            "spam_share": round(float(df.label.mean()), 3),
            "train": len(train),
            "test": len(test),
        },
        "min_precision_target": MIN_PRECISION,
        "deployed": train_baseline(train, test),
    }
    if not args.skip_network:
        metrics["network"] = train_network(train, test)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
