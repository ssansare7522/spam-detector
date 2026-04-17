import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Dataset
data = {
    'text': [
        'Get a free gift card now!', 'Hey, are we meeting?', 
        'WINNER! Claim your prize.', 'The lecture is in APJ lab.',
        'Urgent: Account locked.', 'OTP for transaction is 1234.'
    ],
    'label': [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# 2. Model
model = Pipeline([('vectorizer', CountVectorizer()), ('nb', MultinomialNB())])
model.fit(df['text'], df['label'])

# 3. Website UI
st.title("🚨 AI Spam Guard")
st.write("Developed by: *sumit* | SE Mechanical")

user_msg = st.text_input("Enter message to scan:")
if st.button("Scan"):
    prediction = model.predict([user_msg])[0]
    if prediction == 1:
        st.error("🚨 SPAM ALERT!")
    else:
        st.success("✅ SAFE MESSAGE")
