import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import messagebox

# 1. Internal Dataset (Small for mobile stability)
data = {
    'text': [
        'Get a free gift card now!', 'Hey, are we meeting today?',
        'You won a lottery prize!', 'Please send me the report.',
        'WINNER! Claim your cash now.', 'Call me when you are free.',
        'Urgent: Your account is locked.', 'Lunch tomorrow at 1 PM?'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0] # 1=Spam, 0=Ham
}
df = pd.DataFrame(data)

# 2. Train Model
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
model.fit(df['text'], df['label'])

# 3. Mobile UI logic
def check_message():
    msg = entry.get()
    if not msg:
        return
    
    prediction = model.predict([msg])[0]
    
    if prediction == 1:
        messagebox.showwarning("SPAM ALERT", "🚨 DANGER: This message looks like SPAM!")
    else:
        messagebox.showinfo("SAFE", "✅ This message looks safe.")

# 4. Simple GUI Setup
root = tk.Tk()
root.title("Spam Guard")
root.geometry("400x300")

label = tk.Label(root, text="Enter Message to Scan:", pady=20)
label.pack()

entry = tk.Entry(root, width=40)
entry.pack(pady=10)

btn = tk.Button(root, text="Check for Spam", command=check_message, bg="blue", fg="white")
btn.pack(pady=20)

root.mainloop()
