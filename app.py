import streamlit as st
from inference import predict_fake_news
import google.generativeai as genai
import os

# Configure Gemini
genai.configure(api_key="YOUR API KEY")

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")

st.write("Enter a news article or headline and compare predictions between your trained model and Gemini.")

# Text input
user_input = st.text_area("Paste news text here", height=200)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Model prediction
        model_prediction = predict_fake_news(user_input)

        # Gemini prediction
        prompt = f"Determine if the following news is real or fake. Answer only 'Real' or 'Fake'.\n\nNews: {user_input}"
        gemini_response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        gemini_prediction = gemini_response.text.strip()

        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Your Model", model_prediction)
        with col2:
            st.metric("Gemini", gemini_prediction)
