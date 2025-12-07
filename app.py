import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
nltk.data.path.append('nltk_data')


# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing function
def transform(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    tokens = [t for t in tokens if t not in string.punctuation]
    return " ".join(tokens)

# UI
st.set_page_config(page_title="Spam Detector", page_icon="📡")
st.title("📩 Email/SMS Spam Detector 🚀")
st.write("Enter a message below to detect whether it's **Spam** or **Not Spam**.")

input_sms = st.text_area("Type your message here:", height=120)

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # Preprocess & Vectorize
        transformed_sms = transform(input_sms)
        vector_input = vectorizer.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 This message looks like **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM** 🤝")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
