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

# A sample dataset: past decisions and their possible future outcomes
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

# We flatten the future outcomes to train a basic classifier
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["past_decision"])  # e.g. "started a business"
y_flat = [item for sublist in df["future_outcomes"] for item in sublist]  # flatten possible outcomes

# We'll replicate X to match the length of y_flat
# each row in X is repeated len(future_outcomes[i]) times
repeat_counts = [len(outcomes) for outcomes in df["future_outcomes"]]
X_expanded = np.repeat(X.toarray(), repeat_counts, axis=0)

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
Enter a **past decision** (real or hypothetical) and **describe your current mindset**.  
We'll analyze your sentiment, then generate **multiple possible futures** with probabilities!
""")

# User inputs
past_decision = st.text_input("What past decision have you made?")
mindset_text = st.text_area("Describe your current thoughts and feelings (mindset)")

if st.button("Predict My Future"):
    if not past_decision or not mindset_text:
        st.warning("Please provide both your decision and your mindset.")
    else:
        # 1. Analyze sentiment
        sentiment_score = TextBlob(mindset_text).sentiment.polarity
        mindset = "positive" if sentiment_score > 0 else "negative"

        # 2. Predict outcomes
        decision_vector = vectorizer.transform([past_decision])
        possible_futures = model.predict(decision_vector)  # naive approach returns a single label per row

        # We'll artificially create multiple "future outcomes" from that single prediction
        # to simulate a "multiverse" approach. In reality, you'd have a more complex model.
        # Let's say we create 3 possible futures by combining the predicted outcome with random variants.
        base_future = possible_futures[0]

        # Generate random variants
        variants = [
            f"{base_future} with massive success",
            f"{base_future} but overcame huge obstacles",
            f"{base_future} and discovered a new path"
        ]

        # 3. Assign probabilities based on sentiment
        # For demonstration, if mindset is positive, probabilities shift up, if negative, shift down.
        probabilities = [random.uniform(0.4, 0.9) for _ in variants]
        if mindset == "positive":
            probabilities = [min(p + 0.1, 1.0) for p in probabilities]
        else:
            probabilities = [max(p - 0.1, 0.1) for p in probabilities]

        # 4. Sort by highest probability
        future_predictions = sorted(zip(variants, probabilities), key=lambda x: x[1], reverse=True)

        # 5. Display results
        st.subheader("Your Possible Futures:")
        for i, (future, prob) in enumerate(future_predictions, start=1):
            st.write(f"**{i}.** {future} (Probability: {round(prob * 100, 2)}%)")

        st.info(f"**Sentiment Analysis**: Your mindset is **{mindset}** (score: {round(sentiment_score, 2)})")

st.write("---")
st.write("© 2025 Multiverse AI. For entertainment purposes only.")
