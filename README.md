# 📰 Fake News Detection with BERT  

This project is an **AI-powered Fake News Detector** that classifies news articles as **Fake** or **Real**.  
It uses a **fine-tuned BERT model** trained on the [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and provides an interactive **Streamlit UI** where users can input text and get predictions.  

Additionally, the app integrates with **Google Gemini API** to show Gemini’s prediction for the same news, enabling a side-by-side comparison of your model vs. Gemini.  

---

## ✨ Features
- 🔍 Detects whether a news article is **Fake** or **Real**.  
- 🤖 Uses a **fine-tuned BERT model** for classification.  
- ⚡ Interactive **Streamlit UI** for entering news text.  
- 🔄 Shows predictions from both **your model** and **Gemini API**.  
- 🛡️ Securely manages API keys with `.env` file (ignored in Git).  

---


## 🛠 Installation

1. **Clone the repo from GitHub**  
   ```bash
   git clone https://github.com/Bare009/fake-news-detection-bert.git
   cd fake-news-detection-bert

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   # OR
   source venv/bin/activate   # On Mac/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Gemini API key** in a `.env` file (not included in GitHub, since it’s ignored by `.gitignore`):

   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```


