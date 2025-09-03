📰 Fake News Detection with BERT

This project is an AI-powered Fake News Detector that classifies news articles as Fake or Real.
It uses a fine-tuned BERT model trained on the WELFake dataset
and provides an interactive Streamlit UI where users can input text and get predictions.
Additionally, the app integrates with Google Gemini API to show Gemini’s prediction for the same news, enabling a side-by-side comparison of your model vs. Gemini.

Features:
🔍 Detects whether a news article is Fake or Real.
🤖 Uses a fine-tuned BERT model for classification.
⚡ Interactive Streamlit UI for entering news text.
🔄 Shows predictions from both your model and Gemini API.
🛡️ Securely manages API keys with .env file (ignored in Git).

## 🛠️ Installation
```bash
git clone https://github.com/Bare009/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
