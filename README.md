# 📲 SMS Spam Detector

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44+-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

A deep learning web app that classifies SMS messages as **Spam** or **Non-Spam** in real time, with confidence scoring and session analytics.

**[🚀 Live Demo — Hugging Face Spaces](https://cnoret-sms-spam-detector.hf.space/)** · **[☁️ Live Demo — Streamlit Cloud](https://sms-spam-detector-cnoret.streamlit.app/)**

![App Screenshot](img/screenshot.png)

---

## What I Built

- Trained a **word embedding + Dense** model on the UCI SMS Spam Collection dataset using **TensorFlow / Keras**
- Built an interactive **Streamlit** web app with real-time prediction, confidence visualization, and session history
- Containerized with **Docker** and deployed on both **Hugging Face Spaces** and **Streamlit Cloud**
- Automated syncing between GitHub and Hugging Face with **GitHub Actions**

## Tech Stack

| Layer | Technology |
| --- | --- |
| Deep Learning | TensorFlow 2.18, Keras (Embedding + Dense) |
| Web App | Streamlit 1.44+ |
| Visualization | Plotly |
| Containerization | Docker |
| CI/CD | GitHub Actions → Hugging Face Spaces |

## Features

- Real-time spam detection with confidence score
- Color-coded result card (green / red)
- Random example generator for quick testing
- Prediction history table and proportion chart

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
├── app.py                # Streamlit application
├── models/               # Trained model & tokenizer
├── notebook/             # Training notebook (EDA, model training)
├── Dockerfile
└── requirements.txt
```

## Disclaimer

Trained on ~5,500 messages from the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) (early 2010s, English only). Intended for educational and demonstration purposes — not suitable for production use.

## Author

**Christophe Noret**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/christophenoret/)

## License

[MIT](LICENSE)
