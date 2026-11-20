"""Text preprocessing shared by training and inference.

The model was trained on messages cleaned with ``clean_text`` and padded to
``MAXLEN`` tokens. Any prediction must go through the same steps, otherwise
the scores no longer match the ones measured in the notebook.
"""

import re
from functools import lru_cache

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Length of the padded sequences the model was trained on (input shape (None, 71)).
MAXLEN = 71
NUM_WORDS = 1000
OOV_TOKEN = "<OOV>"
SPACY_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def get_nlp():
    """Load the spaCy pipeline once. Parser and NER do not affect lemmas."""
    return spacy.load(SPACY_MODEL, disable=["parser", "ner"])


def clean_text(text: str) -> str:
    """Lowercase, keep letters only, lemmatize and drop stop words.

    Identical to the ``clean_text`` function used in the training notebook.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"ltgt", "", text)
    text = text.replace("  ", " ").lower().strip()
    doc = get_nlp()(text)
    return " ".join(token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS)


def to_padded_sequences(texts, tokenizer, maxlen: int = MAXLEN):
    """Clean, tokenize and pad a list of raw messages for the model."""
    from tensorflow.keras.utils import pad_sequences

    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    return pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")
