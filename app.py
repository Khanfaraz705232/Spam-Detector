import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing function (NO punkt required)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

def transform(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


# UI
st.set_page_config(page_title="Spam Detector", page_icon="📩")
st.title("📩 Email / SMS Spam Detector 🚀")

input_sms = st.text_area("Enter a message:", height=120)

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first!")
    else:
        transformed_sms = transform(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 SPAM Detected ❌")
        else:
            st.success("🛡 This message is NOT Spam ✔")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Powered by Machine Learning 🤖")
