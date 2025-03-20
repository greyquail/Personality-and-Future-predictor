import streamlit as st
import random
import openai
from textblob import TextBlob

# -----------------------------------------------------------------------------
# 1) Load OpenAI API Key from Streamlit Secrets
# -----------------------------------------------------------------------------
# Ensure you have set your API key in the Streamlit Cloud Secrets as:
# OPENAI_API_KEY = "sk-your_actual_key_here"
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -----------------------------------------------------------------------------
# 2) Initialize Session Memory for Conversation History
# -----------------------------------------------------------------------------
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# -----------------------------------------------------------------------------
# 3) Function to Generate Detailed Explanations Using OpenAI ChatCompletion
# -----------------------------------------------------------------------------
def generate_explanation(choice, future, user_history):
    """
    Uses OpenAI's ChatCompletion API to generate a detailed explanation for why this future might occur,
    based on the user's choice, the predicted future, and the conversation history.
    """
    prompt = f"""
User's choice: {choice}
Possible future: {future}
User's conversation history:
{user_history}

Explain in detail WHY this future might occur, considering the user's mindset and past decisions.
Use a mystical yet logical tone.
"""
    # Use the chat completion endpoint with gpt-3.5-turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a wise and mystical AI oracle."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    explanation = response.choices[0].message.content.strip()
    return explanation

# -----------------------------------------------------------------------------
# 4) Universe Splitter Logic: Predict Multiple Futures
# -----------------------------------------------------------------------------
def predict_multiverse(choice, mindset_score):
    """
    Returns a list of tuples (future_variant, probability) based on the user's choice
    and sentiment (mindset_score). This simulates multiple timeline outcomes.
    For positive sentiment, sample 4 outcomes; for negative, sample 2.
    """
    base_futures = [
        "achieved massive success",
        "discovered an unexpected path",
        "faced enormous challenges",
        "changed the world in a surprising way"
    ]
    if mindset_score > 0:
        possible_futures = random.sample(base_futures, 4)
    else:
        possible_futures = random.sample(base_futures, 2)
    
    futures_with_probs = []
    for fut in possible_futures:
        prob = random.uniform(0.3, 0.9)
        if mindset_score > 0:
            prob = min(prob + 0.1, 1.0)
        else:
            prob = max(prob - 0.1, 0.1)
        futures_with_probs.append((fut, prob))
    
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

        # Append current input to conversation history
        st.session_state["conversation_history"].append(
            f"User choice: {user_choice}, Mindset: {mindset_text}, Sentiment: {sentiment_score}"
        )

        # Generate possible futures using our universe splitter logic
        futures = predict_multiverse(user_choice, sentiment_score)

        st.subheader("Your Possible Futures:")
        for idx, (fut, prob) in enumerate(futures, start=1):
            # Generate explanation using OpenAI GPT via ChatCompletion
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
