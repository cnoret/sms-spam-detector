# SMS Spam Detector
## Overview

This project is a **Streamlit application** that detects whether an SMS message is spam or non-spam using a pre-trained deep learning model. The model was trained on SMS data and uses natural language processing techniques for prediction. The application provides an intuitive interface for users to input messages and view predictions.

![App Screenshot](/img/screenshot.png)

## Features

- **Interactive Interface:** Enter SMS messages to classify them as spam or non-spam.
- **Real-time Predictions:** Displays the predicted label along with the confidence score.
- **Visualization:** Pie chart showing the proportion of spam vs non-spam predictions.
- **Prediction History:** Maintains a log of all predictions made during the session.
- **Custom Styling:** User-friendly interface with CSS enhancements.

## Demo

You can try the application live:
- **Docker Deployment on Hugging Face Spaces:** [SMS Spam Detector - Docker](https://cnoret-sms-spam-detector.hf.space/)
- **Streamlit Cloud Deployment:** [SMS Spam Detector - Streamlit](https://sms-spam-detector-cnoret.streamlit.app/)

## Installation

### Prerequisites

- Python 3.8 or later
- pip

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/sms-spam-detector.git
   cd sms-spam-detector
   ```

2. Create a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model and tokenizer:

   - Place the `model_wordembed.keras` file in the `models/` directory.
   - Place the `tokenizer_word_index.npy` file in the same directory.

5. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

6. Access the app in your browser at `http://localhost:8501`.

## Project Structure

```
├── app.py                # Main Streamlit application
├── models/               # Directory for the model and tokenizer
│   ├── model_wordembed.keras
│   ├── tokenizer_word_index.npy
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Example Usage

1. Start the application by running `streamlit run app.py`.
2. Enter a message in the text box.
3. Click the "Predict" button to view the classification result.
4. Check the visualization for the proportion of predictions.

## Technologies Used

- **Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** TensorFlow, Keras
- **Visualization:** Plotly
- **NLP:** Tokenizer, Embedding layers

## Future Enhancements

- Add support for additional languages.
- Include training scripts for fine-tuning the model.
- Enhance visualizations with detailed analytics.
- Allow users to choose between multiple pre-trained models within the application.


## Author

**Christophe Noret**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/christophenoret/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Dependencies Licenses

- **Streamlit:** Licensed under the Apache 2.0 License. For details, see [Streamlit's GitHub repository](https://github.com/streamlit/streamlit/blob/develop/LICENSE).
- **TensorFlow:** Licensed under the Apache 2.0 License. For details, see [TensorFlow's license](https://github.com/tensorflow/tensorflow/blob/master/LICENSE).
- **Plotly:** Licensed under the MIT License. For details, see [Plotly's GitHub repository](https://github.com/plotly/plotly.py/blob/master/LICENSE.txt).

