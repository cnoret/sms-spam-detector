# SMS Spam Detector: A Deep Learning Spam Filter, Checked Against a Baseline

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://sms-spam-detector-cnoret.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cnoret/sms-spam-detector/blob/main/notebook/AT%26T_spam_detector.ipynb)
[![Tests](https://github.com/cnoret/sms-spam-detector/actions/workflows/tests.yml/badge.svg)](https://github.com/cnoret/sms-spam-detector/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**A Keras network reading both words and characters lifts spam F1 from 0.856 to 0.964 under a leak-free evaluation, with a single legitimate SMS flagged on 1,032 held-out messages, on par with a TF-IDF + logistic regression baseline (0.972).**

`Data:` 5,572 SMS · 5,158 unique after deduplication · 12.4% spam · English, UCI SMS Spam Collection (early 2010s)

`Live app:` [Streamlit Cloud](https://sms-spam-detector-cnoret.streamlit.app/) · [Hugging Face Spaces](https://cnoret-sms-spam-detector.hf.space/)

![App Screenshot](https://raw.githubusercontent.com/cnoret/sms-spam-detector/main/img/screenshot.png)

---

## What I Built

- Designed a **Keras** network that reads raw SMS through two branches, **word embeddings** and a **character-level CNN**, with the text vectorization saved inside the model
- Chose the architecture with **5-fold cross-validation on the train set only**, out of six candidates (BiGRU, character CNNs and their combinations)
- Set up a leak-free evaluation (deduplication, stratified split, threshold chosen out of fold) in a reproducible `train.py`, shared by every model
- Benchmarked the network against a **TF-IDF + logistic regression** baseline and against the first version of the network, on the same held-out messages
- Built an interactive **Streamlit** web app where you pick the model and see both models' scores side by side
- Containerized with **Docker** and deployed on both **Hugging Face Spaces** and **Streamlit Cloud**
- Added **pytest** and **Streamlit AppTest** checks, run with **ruff** on every push and pull request with **GitHub Actions**
- Automated syncing between GitHub and Hugging Face with **GitHub Actions**

## Tech Stack

| Layer | Technology |
| --- | --- |
| Neural network | TensorFlow 2.21, Keras 3 (TextVectorization, embeddings, Conv1D) |
| Baseline | scikit-learn 1.7 (TF-IDF + logistic regression) |
| First network (reference) | Keras (Embedding + Dense), spaCy |
| Web App | Streamlit 1.44+ |
| Visualization | Plotly |
| Containerization | Docker |
| Tests | pytest, Streamlit AppTest, ruff |
| CI/CD | GitHub Actions (tests on every push and PR, sync to Hugging Face Spaces) |

## Data Insights

A few findings from the exploratory analysis (5,572 messages, 87% ham / 13% spam):

**Duplicates** — 414 rows are exact copies of another message (5,158 unique messages). Left in, they put copies of training messages in the test set and inflate the scores, so they are removed before splitting.

**Class imbalance** — The 87/13 split is handled with balanced class weights, and accuracy is not used to pick the model: a model that always answers "non-spam" would already reach 87%.

**Spam messages are nearly twice as long** — Average length: 142 characters for spam vs 74 for ham. Message length alone is a strong signal.

**Most frequent words in spam** — `free`, `txt`, `win`, `claim`, `mobile`, `stop`, `reply`. Classic urgency/reward vocabulary that differs clearly from everyday messages.

## Model

**Protocol** — Duplicates removed, one stratified 80/20 split (4,126 train / 1,032 test), test set used once. Every choice (architecture, vocabulary, number of epochs, decision threshold) is made on the train set. The threshold is the one that keeps spam precision at 98% or more on out-of-fold predictions: blocking a legitimate SMS costs more than letting a spam through.

**Neural network** — The model takes the raw message and splits it two ways:

```text
raw SMS ─┬─ word vectorizer (10k words, 64 tokens) → embedding 64 → global max pooling ──┐
         └─ character vectorizer (300 chars) → embedding 32 → Conv1D 128×5 → max pooling ─┴─ concat → dropout 0.3 → dense 32 → sigmoid
```

The word branch picks up the vocabulary of spam (`free`, `claim`, `prize`), the character branch the spellings that word tokenizers lose (`£1000`, `txt`, phone numbers, short codes). The two vectorizers are layers of the model, learned on the train set and saved with it, so the app sends the raw text and there is no separate preprocessing to keep in sync. 673k parameters, 2.8 MB. Training uses balanced class weights and early stopping on each validation fold; the final model is trained on the whole train set for the median number of epochs (9).

**Architecture selection** — Spam F1 on out-of-fold predictions of the train set, at the 98% precision threshold:

| Candidate | CV spam F1 |
| --- | --- |
| Word embeddings + BiGRU | 0.901 |
| Character CNN + BiGRU | 0.880 |
| Character CNN, kernels 3/5/7 | 0.922 |
| Character CNN, kernel 5 | 0.932 |
| Word BiGRU + character CNN | 0.940 |
| **Word embeddings (max pooling) + character CNN** | **0.953** |
| TF-IDF + logistic regression (baseline) | 0.962 |

**Baseline** — TF-IDF on character 2–5-grams + logistic regression with balanced class weights, 834 KB. It is the point of comparison: a strong linear model on character n-grams shows whether the network learns more than n-gram counts.

**Results on the 1,032 held-out unique messages** (128 spams)

| Model | Spam precision | Spam recall | Spam F1 | Legitimate SMS flagged | Spams missed |
| --- | --- | --- | --- | --- | --- |
| **Word + character CNN (Keras)** | **0.992** | **0.938** | **0.964** | **1** | **8** |
| TF-IDF + logistic regression (baseline) | 0.984 | 0.961 | 0.972 | 2 | 5 |
| First network: Embedding + Dense, spaCy cleaning | 0.970 | 0.766 | 0.856 | 3 | 30 |

**Reading the results** — The redesigned network catches 22 more spams than the first version at the same precision target. Against the baseline it is a tie on this test set: one fewer false alarm, three more spams missed, a gap of a few messages out of 128 spams that the test set is too small to separate. The network is the heavier option to serve (TensorFlow, about 800 MB of memory at runtime), which is the price of a model that learns its own text representation; the baseline stays in the app to make that comparison visible.

All numbers are written by `train.py` to [`models/metrics.json`](models/metrics.json), which the app reads.

**What changed from the first version** — The original notebook reported F1 0.89 for the network, but on a test set where 10% of messages were also in the training data, and the deployed app skipped the spaCy cleaning used in training, which brought its real spam precision down to 0.54. The network is now evaluated on deduplicated data, and its preprocessing lives inside the saved model, so training and serving cannot drift apart.

## Retrain

```bash
pip install -r requirements-train.txt
python train.py                  # network, baseline and first network (about 3 minutes on CPU)
python train.py --skip-original  # no spaCy needed
python train.py --skip-network   # baseline only
```

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
python -m pytest
```

The suite (about 20 seconds) checks the training protocol, retrains the baseline to confirm it reproduces `models/metrics.json`, checks that the network saves and loads back with its vocabulary, verifies both committed models on known spam and everyday messages, and drives the Streamlit app end to end with AppTest (model switch included). `python -m pytest --run-slow` also retrains the network (about 2 minutes) and compares it with the committed metrics.

## Run Locally

**Requires:** Docker

```bash
git clone https://github.com/cnoret/sms-spam-detector.git
cd sms-spam-detector
docker build -t sms-spam-detector .
docker run -p 7860:7860 sms-spam-detector
```

Open `http://localhost:7860`

## Project Structure

```text
├── app.py                 # Streamlit application
├── train.py               # Training and evaluation of all models (writes models/)
├── preprocessing.py       # spaCy cleaning for the first network (reference)
├── models/                # Network (.keras), baseline (.joblib), metrics.json
├── notebook/              # Exploratory notebook (EDA, first network)
├── tests/                 # pytest + Streamlit AppTest
├── .github/workflows/     # Tests, sync to Hugging Face
├── Dockerfile
├── ruff.toml
├── requirements.txt       # App
├── requirements-dev.txt   # App + test tools
└── requirements-train.txt # Training
```

## Disclaimer

Trained on ~5,500 messages from the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) (early 2010s, English only). Modern phishing SMS (fake bank or parcel links) are under-represented and some slip through. Intended for educational and demonstration purposes — not suitable for production use.

## Author

**Christophe Noret**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/christophenoret/)

## License

[MIT](LICENSE)
