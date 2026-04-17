import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import time

# 1. Page Config
st.set_page_config(page_title="AI Auto-Scan Guard", page_icon="🚨")

# 2. Dataset & Training
data = {
    'text': [
        'Get a free gift card!', 'Meeting at 5pm', 'WINNER! Claim cash',
        'Your OTP is 1234', 'Account locked: click here', 'Lunch tomorrow?'
    ],
    'label': [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
model = Pipeline([('vectorizer', CountVectorizer()), ('nb', MultinomialNB())])
model.fit(df['text'], df['label'])

# 3. Website UI
st.title("🚨 AI Spam Guard (Auto-Mode)")
st.write("Developed by: *Sumit Sansare* | SE Mechanical")

# The Auto-Scan logic for the Demo
st.subheader("Simulated Auto-Scan")
st.info("Copy any message from your SMS inbox, then click the 'Check Clipboard' button below.")

if st.button("Check Clipboard & Auto-Scan"):
    # Note: Streamlit uses browser-based clipboard access
    st.write("Reading message from device memory...")
    # In a real demo, we use a text area that the user can quickly 'Paste' into 
    # as a backup because browsers protect clipboard security.
    test_input = st.text_area("Live Message Feed:", placeholder="Paste or Copy message here...")
    
    if test_input:
        prediction = model.predict([test_input])[0]
        if prediction == 1:
            st.error("🚨 SCAM DETECTED! This message is dangerous.")
        else:
            st.success("✅ MESSAGE SAFE. No threats found.")
