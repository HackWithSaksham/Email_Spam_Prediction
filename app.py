import streamlit as st
import joblib

# =========================
# Load Model & Vectorizer
# =========================
model = joblib.load("fake_email_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# UI
# =========================
st.set_page_config(page_title="Fake Email Detector", page_icon="📧")

st.title("📧 Fake Email Detection System")
st.write("Detect whether an email is **Fake (Spam/Phishing)** or **Real (Safe)**.")

# =========================
# Input
# =========================
email_text = st.text_area("✍️ Enter Email Content:")

# =========================
# Prediction
# =========================
if st.button("🔍 Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some email text!")
    else:
        # Transform input
        text_vector = vectorizer.transform([email_text])

        # Predict
        prediction = model.predict(text_vector)[0]
        prob = model.predict_proba(text_vector)[0]

        # =========================
        # Output
        # =========================
        if prediction == 1:
            st.error(f"🚨 Fake Email Detected!")
            st.write(f"Confidence: {max(prob)*100:.2f}%")
        else:
            st.success(f"✅ Real Email")
            st.write(f"Confidence: {max(prob)*100:.2f}%")