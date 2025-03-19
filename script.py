import streamlit as st
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

# ------------------------------
# 1) DATASET & MODEL PREPARATION
# ------------------------------

# Sample dataset: past decisions and their possible future outcomes
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

# Flatten the future outcomes to train a basic classifier
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["past_decision"])  # e.g. "started a business"
# For each row in X, repeat it as many times as there are outcomes in that row.
repeat_counts = [len(outcomes) for outcomes in df["future_outcomes"]]
X_expanded = np.repeat(X.toarray(), repeat_counts, axis=0)
y_flat = [outcome for outcomes in df["future_outcomes"] for outcome in outcomes]

# Train a simple Naive Bayes model
model = MultinomialNB()
model.fit(X_expanded, y_flat)

# --------------------------------
# 2) STREAMLIT APP (FRONTEND + LOGIC)
# --------------------------------

st.set_page_config(page_title="Multiverse Future Predictor", layout="centered")

st.title("🔮 Multiverse Future Predictor")
st.write("""
Welcome to the **Multiverse Future Predictor**!  
Enter a **past decision** and describe your **current mindset**.  
We analyze your sentiment and generate multiple possible futures with probabilities!
""")

# User inputs
past_decision = st.text_input("Enter a past decision (e.g., 'started a business')")
mindset_text = st.text_area("Describe your current thoughts and feelings")

if st.button("Predict My Future"):
    if not past_decision or not mindset_text:
        st.warning("Please provide both your decision and your mindset.")
    else:
        # Analyze sentiment using TextBlob
        sentiment_score = TextBlob(mindset_text).sentiment.polarity
        mindset = "positive" if sentiment_score > 0 else "negative"

        # Predict base outcome using our model
        decision_vector = vectorizer.transform([past_decision])
        predicted_outcome = model.predict(decision_vector)[0]

        # Generate multiple "multiverse" variants
        variants = [
            f"{predicted_outcome} with massive success",
            f"{predicted_outcome} but faced enormous challenges",
            f"{predicted_outcome} and discovered an unexpected path"
        ]

        # Assign probabilities (adjust based on sentiment)
        probabilities = [random.uniform(0.4, 0.8) for _ in variants]
        if mindset == "positive":
            probabilities = [min(p + 0.1, 1.0) for p in probabilities]
        else:
            probabilities = [max(p - 0.1, 0.1) for p in probabilities]

        # Sort results by probability descending
        future_predictions = sorted(zip(variants, probabilities), key=lambda x: x[1], reverse=True)

        # Display results
        st.subheader("Your Possible Futures:")
        for i, (future, prob) in enumerate(future_predictions, start=1):
            st.write(f"**{i}.** {future} (Probability: {round(prob * 100, 2)}%)")

        st.info(f"Sentiment Score: {round(sentiment_score, 2)} - Your mindset is **{mindset}**")

st.write("---")
st.write("© 2025 Multiverse AI. For entertainment purposes only.")
