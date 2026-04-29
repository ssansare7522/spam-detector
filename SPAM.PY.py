import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Page Configuration
st.set_page_config(page_title="AI Spam Guard", page_icon="🛡️")
st.title("🛡️ AI Spam Message Detector")
st.write("Project by: SE Mechanical - AVCOE")

# 2. Balanced Dataset - This fixes the 'Always Dangerous' bug!
data = {
    "text": [
        # --- DANGEROUS (SPAM) ---
        "Get a free gift card now!", "WINNER! You won a prize", 
        "URGENT: Account hacked", "Congratulations lottery winner",
        "Verify your OTP now", "Claim your cash reward", "Click here for money",
        
        # --- SAFE (HAM) ---
        "Hello, how are you?", "The mechanical lab is open", 
        "Submit the report by tomorrow", "Meeting with Dagale Sir at 10",
        "See you at the college library", "Please mark my attendance",
        "Can we discuss the project?", "I am coming to Sangamner",
        "Practical exams start next week", "Good morning team"
    ],
    "label": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
}

# 3. Build the AI Pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Train the model instantly
model.fit(data["text"], data["label"])

# 4. User Interface
user_input = st.text_area("Paste the message here:", placeholder="Type something like 'Hello Sir'...")

if user_input:
    prediction = model.predict([user_input])[0]
    
    if prediction == 1:
        st.error("🚨 DANGEROUS: This message looks like a Scam/Spam!")
    else:
        st.success("✅ SAFE: This message appears to be legitimate.")

st.info("Algorithm used: Multinomial Naive Bayes")
