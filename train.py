"""Train the SMS spam models and write their evaluation to models/metrics.json.

Protocol (fixes the leaks found in the original notebook), shared by every model:
- messages are deduplicated before splitting (414 duplicate rows in the raw file),
- one stratified 80/20 train/test split, the test set is used once at the end,
- the decision threshold is chosen on out-of-fold predictions of the train set,
- everything a model learns (vocabulary, weights, epochs) comes from the train set only.

Models:
- the neural network studied here: word embeddings + a character CNN, with the
  text vectorization inside the Keras model so it takes raw messages,
- the comparison baseline: TF-IDF on character n-grams + logistic regression,
- the original Embedding + Dense network (spaCy cleaning), kept as a reference.

Usage:
    python train.py                  # all three models
    python train.py --skip-original  # no spaCy needed
    python train.py --skip-network   # baseline only (no TensorFlow needed)
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
NETWORK_PATH = Path("models/spam_network.keras")
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
        "cv_f1": round(f1_score(train.label, oof >= threshold), 3),
        "test": scores(test.label, (proba >= threshold).astype(int)),
        "test_at_0.5": scores(test.label, (proba >= 0.5).astype(int)),
        "size_kb": round(MODEL_PATH.stat().st_size / 1024),
    }


# Network hyperparameters, picked with 5-fold cross-validation on the train set
# only (word BiGRU, character CNNs and their combinations were compared).
WORD_VOCAB = 10_000
WORD_LEN = 64
CHAR_VOCAB = 200
CHAR_LEN = 300
MAX_EPOCHS = 30


def _adapt(vectorizer, texts) -> None:
    """Learn the vocabulary, minus line breaks.

    Keras stores the vocabulary one token per line: a "\\r" or "\\n" token would
    corrupt the saved model. Dropped tokens map to the OOV index, at training
    and at prediction time alike.
    """
    vectorizer.adapt(texts)
    vocab = vectorizer.get_vocabulary(include_special_tokens=False)
    vectorizer.set_vocabulary([t for t in vocab if "\n" not in t and "\r" not in t])


def build_network(texts):
    """Word embeddings + character CNN on raw text.

    The two TextVectorization layers are adapted on ``texts`` and saved inside
    the model, so the app feeds it the raw message with no separate tokenizer.
    """
    import keras
    from keras import layers

    inp = keras.Input(shape=(1,), dtype="string", name="message")

    words = layers.TextVectorization(max_tokens=WORD_VOCAB, output_sequence_length=WORD_LEN)
    _adapt(words, texts)
    w = layers.Embedding(WORD_VOCAB, 64)(words(inp))
    w = layers.GlobalMaxPooling1D()(w)

    chars = layers.TextVectorization(
        split="character", standardize="lower", max_tokens=CHAR_VOCAB,
        output_sequence_length=CHAR_LEN,
    )
    _adapt(chars, texts)
    c = layers.Embedding(CHAR_VOCAB, 32)(chars(inp))
    c = layers.Conv1D(128, 5, activation="relu")(c)
    c = layers.GlobalMaxPooling1D()(c)

    x = layers.Concatenate()([w, c])
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="spam_probability")(x)

    model = keras.Model(inp, out, name="word_char_cnn")
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def _as_tensor(messages):
    import tensorflow as tf

    return tf.constant(list(messages))[:, None]


def _class_weight(labels) -> dict:
    labels = np.asarray(labels)
    return {k: len(labels) / (2 * int((labels == k).sum())) for k in (0, 1)}


def network_proba(model, messages) -> np.ndarray:
    return model.predict(_as_tensor(messages), verbose=0).ravel()


def train_network(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Neural network under the baseline protocol.

    5-fold cross-validation on the train set gives the out-of-fold predictions
    used to pick the threshold, and the number of epochs (early stopping on each
    validation fold). The final model is then fitted on the whole train set.
    """
    import keras
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    x, y = train.message.to_numpy(dtype=object), train.label.to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    best_epochs = []
    for fit_idx, val_idx in cv.split(x, y):
        keras.utils.set_random_seed(SEED)
        model = build_network(x[fit_idx])
        history = model.fit(
            _as_tensor(x[fit_idx]),
            y[fit_idx],
            validation_data=(_as_tensor(x[val_idx]), y[val_idx]),
            epochs=MAX_EPOCHS,
            batch_size=32,
            class_weight=_class_weight(y[fit_idx]),
            callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=0,
        )
        best_epochs.append(int(np.argmin(history.history["val_loss"])) + 1)
        oof[val_idx] = network_proba(model, x[val_idx])
    threshold = choose_threshold(y, oof)
    epochs = int(np.median(best_epochs))

    keras.utils.set_random_seed(SEED)
    model = build_network(x)
    model.fit(
        _as_tensor(x), y, epochs=epochs, batch_size=32,
        class_weight=_class_weight(y), verbose=0,
    )
    proba = network_proba(model, test.message)

    # Inference copy without the optimizer state, to keep the file small.
    NETWORK_PATH.parent.mkdir(exist_ok=True)
    keras.Model(model.inputs, model.outputs, name=model.name).save(NETWORK_PATH)

    return {
        "name": "Word embeddings + character CNN (Keras)",
        "threshold": round(threshold, 3),
        "cv_f1": round(f1_score(y, oof >= threshold), 3),
        "epochs": epochs,
        "parameters": int(model.count_params()),
        "test": scores(test.label, (proba >= threshold).astype(int)),
        "test_at_0.5": scores(test.label, (proba >= 0.5).astype(int)),
        "size_kb": round(NETWORK_PATH.stat().st_size / 1024),
    }


def train_original_network(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Original Embedding + Dense network (spaCy cleaning), kept as a reference."""
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
    parser.add_argument("--skip-network", action="store_true", help="baseline only")
    parser.add_argument("--skip-original", action="store_true", help="no spaCy reference")
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
    }
    if not args.skip_network:
        import tensorflow as tf

        metrics["tensorflow"] = tf.__version__
        metrics["network"] = train_network(train, test)
    metrics["baseline"] = train_baseline(train, test)
    if not (args.skip_network or args.skip_original):
        metrics["original_network"] = train_original_network(train, test)

    # Keep the entries of the models that were not retrained this time.
    if METRICS_PATH.exists():
        previous = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        for key in ("tensorflow", "network", "original_network"):
            if key not in metrics and key in previous:
                metrics[key] = previous[key]

    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
