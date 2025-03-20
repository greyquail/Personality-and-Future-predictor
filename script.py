import streamlit as st
import random
import openai
from textblob import TextBlob

# -----------------------------------------------------------------------------
# 1) Load OpenAI API Key from Streamlit Secrets
# -----------------------------------------------------------------------------
# Ensure you have added your key in the Streamlit Cloud Secrets (as shown above)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -----------------------------------------------------------------------------
# 2) Initialize Session Memory for Conversation History
# -----------------------------------------------------------------------------
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# -----------------------------------------------------------------------------
# 3) Function to Generate Detailed Explanations Using OpenAI
# -----------------------------------------------------------------------------
def generate_explanation(choice, future, user_history):
    """
    Uses OpenAI's GPT to generate a detailed explanation for why this future might occur,
    based on the user's choice, the predicted future, and the conversation history.
    """
    prompt = f"""
You are a wise, futuristic AI oracle with deep insight into multiple timelines.
User's choice: {choice}
Possible future: {future}
User's conversation history:
{user_history}

Explain in detail WHY this future might occur, considering the user's mindset and past decisions.
Use a mystical yet logical tone.
"""
    response = openai.Completion.create(
        engine="text-davinci-003",  # Alternatively, use "gpt-3.5-turbo" if preferred.
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )
    explanation = response["choices"][0]["text"].strip()
    return explanation

# -----------------------------------------------------------------------------
# 4) Universe Splitter Logic: Predict Multiple Futures
# -----------------------------------------------------------------------------
def predict_multiverse(choice, mindset_score):
    """
    Returns a list of tuples (future_variant, probability) based on the user's choice
    and sentiment (mindset_score). This simulates multiple timeline outcomes.
    """
    base_futures = [
        "achieved massive success",
        "discovered an unexpected path",
        "faced enormous challenges",
        "changed the world in a surprising way"
    ]
    # Adjust outcomes based on sentiment
    if mindset_score > 0:
        positive_futures = random.sample(base_futures[:2], 2)
        negative_futures = random.sample(base_futures[2:], 2)
    else:
        positive_futures = random.sample(base_futures[:2], 1)
        negative_futures = random.sample(base_futures[2:], 3)
    possible_futures = positive_futures + negative_futures
    # Assign random probabilities and adjust based on sentiment
    futures_with_probs = []
    for fut in possible_futures:
        prob = random.uniform(0.3, 0.9)
        if mindset_score > 0:
            prob = min(prob + 0.1, 1.0)
        else:
            prob = max(prob - 0.1, 0.1)
        futures_with_probs.append((fut, prob))
    # Sort by probability descending
    futures_with_probs.sort(key=lambda x: x[1], reverse=True)
    return futures_with_probs

# -----------------------------------------------------------------------------
# 5) Streamlit App UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Multiverse Future Predictor", layout="centered")
st.title("🔮 Advanced Multiverse Future Predictor + LLM")
st.write("""
Welcome to your AI oracle. Enter a past decision or question and describe your current mindset.
Our AI will analyze your sentiment, predict multiple possible futures, and provide detailed explanations
for why these outcomes might occur.
""")

# User input fields
user_choice = st.text_input("Enter a past decision or question")
mindset_text = st.text_area("Describe your current mindset or situation")

if st.button("Predict My Future"):
    if not user_choice or not mindset_text:
        st.warning("Please provide both your decision and your mindset.")
    else:
        # Analyze sentiment using TextBlob
        sentiment_score = TextBlob(mindset_text).sentiment.polarity

        # Save current input to conversation history
        st.session_state["conversation_history"].append(
            f"User choice: {user_choice}, Mindset: {mindset_text}, Sentiment: {sentiment_score}"
        )

        # Generate multiple possible futures
        futures = predict_multiverse(user_choice, sentiment_score)

        st.subheader("Your Possible Futures:")
        for idx, (fut, prob) in enumerate(futures, start=1):
            # Generate explanation using OpenAI
            explanation = generate_explanation(
                choice=user_choice,
                future=fut,
                user_history="\n".join(st.session_state["conversation_history"])
            )
            st.write(f"**{idx}.** {fut} (Probability: {round(prob * 100, 2)}%)")
            st.write(f"*Explanation:* {explanation}")
            st.write("---")

        # Display sentiment information
        if sentiment_score > 0:
            mindset_label = "positive"
        elif sentiment_score < 0:
            mindset_label = "negative"
        else:
            mindset_label = "neutral"
        st.info(f"Sentiment Score: {round(sentiment_score, 2)} - Your mindset is **{mindset_label}**")

st.write("© 2025 Multiverse AI. For entertainment purposes only.")
