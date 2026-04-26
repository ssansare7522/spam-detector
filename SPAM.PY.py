import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Page Configuration
st.set_page_config(page_title="AI Spam Detector", page_icon="🚨")

# 2. Dataset & Internal Training Logic
# This ensures the model works even if you don't upload a separate CSV
def load_model():
    data = {
        'text': [
            'Get a free gift card now', 'Meeting at 5pm today', 'WINNER! Claim your cash prize',
            'Your OTP is 1234', 'Account locked: click here to verify', 'Lunch tomorrow?',
            'The mechanical lab is open', 'Your bank account is hacked', 'Verify your identity',
            'Congratulations! You won a lottery', 'Can we reschedule the meeting?', 
            'Urgent: Your credit card has been suspended', 'Call me back later'
        ],
        'label': [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    # Pipeline: Vectorizer + Naive Bayes
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])
    model.fit(df['text'], df['label'])
    return model

model = load_model()

# 3. User Interface (UI)
st.title("🚨 AI Spam Message Detector")
st.markdown("### *Department of Mechanical Engineering, AVCOE*")
st.write("Project by: *Sumit Sansare* (Roll No: 425)")

st.divider()

# 4. Input Section
st.subheader("Message Analysis")
user_input = st.text_area("Paste or Type the message below:", placeholder="Waiting for input...")

# 5. Instant Detection Logic
if user_input.strip() != "":
    # Perform prediction
    prediction = model.predict([user_input])[0]
    
    st.write("---")
    st.subheader("System Diagnostic Result:")
    
    if prediction == 1:
        st.error("🚨 *DANGER: SPAM / SCAM DETECTED!*")
        st.info("*AI Insight:* This message contains patterns frequently found in fraudulent or phishing communications.")
    else:
        st.success("✅ *SAFE: CLEAN MESSAGE*")
        st.info("*AI Insight:* This message appears to be legitimate and safe to read.")
else:
    st.info("The AI is ready. Please enter a message above to start the real-time scan.")

# 6. Footer
st.divider()
st.caption("Industry 4.0 Mini-Project | Guided by: Prof. Dagale Sir")
