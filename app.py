import streamlit as st
import re
import string
import joblib

# Caching the model and vectorizer to avoid reloading on every interaction
@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.joblib")

@st.cache_resource
def load_model():
    return joblib.load("calibrated_lr_model.joblib")

# Load the vectorizer and model
vectorizer = load_vectorizer()
model = load_model()

# Function to clean the text input
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit app layout
st.title("📰 Fake News Detection System")
st.write("🔍 Enter a news article below to check if it is **Real or Fake** using Machine Learning.")

user_input = st.text_area("📝 News Article")

if st.button("Check News"):
    if not user_input.strip():
        st.error("⚠️ Please enter some news text to check.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        if prediction == 0:
            st.success("🟢 This news is predicted to be **REAL**.")
        else:
            st.error("🔴 This news is predicted to be **FAKE**.")

        # Show prediction probabilities for better interpretability
        st.markdown("### 🔍 Prediction Details:")
        st.write(f"**Probability Real:** {proba[0]:.4f}")
        st.write(f"**Probability Fake:** {proba[1]:.4f}")
        st.write("**Cleaned Input:**", cleaned)
