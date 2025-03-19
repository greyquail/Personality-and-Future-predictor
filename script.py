import streamlit as st
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import nltk

# Download necessary NLTK corpora for TextBlob
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Dataset & Model Preparation
data = {
    "past_decision": [
        "started a business", "invested in stocks", "studied AI", 
        "moved to a new city", "took a risk in career", "stayed in comfort zone", 
        "chose innovation", "ignored learning", "followed passion", "became spiritual"
    ],
    "future_outcomes": [
        ["became successful", "faced struggles", "built a company"],
        ["made profits", "lost money", "became an investor"],
        ["built AI projects", "became a researcher", "created new technology"],
        ["adapted quickly", "felt lost", "grew emotionally"],
        ["achieved success", "faced failure", "gained wisdom"],
        ["life remained the same", "missed opportunities", "felt safe"],
        ["changed the world", "struggled with execution", "created a legacy"],
        ["stagnated", "regretted later", "recovered slowly"],
        ["fulfilled dreams", "encountered big hurdles", "found true happiness"],
        ["attained peace", "discovered hidden truths", "inspired others"]
    ]
}

df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["past_decision"])
y_flat = [item for sublist in df["future_outcomes"] for item in sublist]
repeat_counts = [len(outcomes) for outcomes in df["future_outcomes"]]
X_expanded = np.repeat(X.toarray(), repeat_counts, axis=0)
model = MultinomialNB()
model.fit(X_expanded, y_flat)

# Streamlit App
st.set_page_config(page_title="Multiverse Future Predictor", layout="centered")
st.title("🔮 Multiverse Future Predictor")
st.write("""
Welcome! Enter a past decision and describe your current mindset.
We will simulate multiple possible futures for you.
""")

past_decision = st.text_input("Enter your past decision:")
mindset_text = st.text_area("Describe your current thoughts and feelings:")

if st.button("Predict My Future"):
    if not past_decision or not mindset_text:
        st.warning("Please provide both a decision and your mindset.")
    else:
        sentiment_score = TextBlob(mindset_text).sentiment.polarity
        mindset = "positive" if sentiment_score > 0 else "negative"
        decision_vector = vectorizer.transform([past_decision])
        possible_futures = model.predict(decision_vector)
        base_future = possible_futures[0]
        variants = [
            f"{base_future} with massive success",
            f"{base_future} but overcame huge obstacles",
            f"{base_future} and discovered a new path"
        ]
        probabilities = [random.uniform(0.4, 0.9) for _ in variants]
        if mindset == "positive":
            probabilities = [min(p + 0.1, 1.0) for p in probabilities]
        else:
            probabilities = [max(p - 0.1, 0.1) for p in probabilities]
        future_predictions = sorted(zip(variants, probabilities), key=lambda x: x[1], reverse=True)
        st.subheader("Your Possible Futures:")
        for i, (future, prob) in enumerate(future_predictions, start=1):
            st.write(f"**{i}.** {future} (Probability: {round(prob * 100, 2)}%)")
        st.info(f"Sentiment: Your mindset is {mindset} (score: {round(sentiment_score, 2)})")
st.write("---")
st.write("© 2025 Multiverse AI")
