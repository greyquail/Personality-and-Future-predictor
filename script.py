import streamlit as st
import random
import openai
from textblob import TextBlob

# -------------------------------------------
# 1) Configure OpenAI
# -------------------------------------------
# Replace with your actual OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# -------------------------------------------
# 2) Maintain Conversation Memory
# -------------------------------------------
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# -------------------------------------------
# 3) LLM Explanation Function
# -------------------------------------------
def generate_explanation(choice, future, user_history):
    """
    Uses OpenAI GPT to generate a rationale for why the user
    might have this future, based on their choice, sentiment,
    and conversation history.
    """
    prompt = f"""
You are a wise, futuristic AI that can see multiple timelines.
User's choice: {choice}
Possible future: {future}
User's past context: {user_history}

Explain in detail WHY this future might occur and how the user's mindset influences it.
Use a mystical yet logical style.
"""
    response = openai.Completion.create(
        engine="text-davinci-003",  # or "gpt-3.5-turbo" with ChatCompletion
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )
    explanation = response["choices"][0]["text"].strip()
    return explanation

# -------------------------------------------
# 4) Universe Splitter Logic
# -------------------------------------------
def predict_multiverse(choice, mindset_score):
    """
    Returns a list of (future_variant, probability).
    We'll adjust probability by the mindset score.
    """
    # Example base futures
    base_futures = [
        "achieved massive success",
        "discovered an unexpected path",
        "faced enormous challenges",
        "changed the world in a surprising way"
    ]

    # Weighted random approach
    if mindset_score > 0:
        positive_futures = random.sample(base_futures[:2], 2)
        negative_futures = random.sample(base_futures[2:], 2)
    else:
        positive_futures = random.sample(base_futures[:2], 1)
        negative_futures = random.sample(base_futures[2:], 3)

    possible_futures = positive_futures + negative_futures

    # Assign random probabilities
    futures_with_probs = []
    for fut in possible_futures:
        prob = random.uniform(0.3, 0.9)
        # shift prob if mindset is positive or negative
        if mindset_score > 0:
            prob = min(prob + 0.1, 1.0)
        else:
            prob = max(prob - 0.1, 0.1)
        futures_with_probs.append((fut, prob))

    # Sort by probability descending
    futures_with_probs.sort(key=lambda x: x[1], reverse=True)
    return futures_with_probs

# -------------------------------------------
# 5) Streamlit App
# -------------------------------------------
st.title("🔮 Advanced Multiverse Future Predictor + LLM")

user_choice = st.text_input("Enter a past decision or question")
mindset_text = st.text_area("Describe your current mindset or situation")

if st.button("Predict My Future"):
    if not user_choice or not mindset_text:
        st.warning("Please provide both your decision and mindset.")
    else:
        # 1. Analyze sentiment
        sentiment_score = TextBlob(mindset_text).sentiment.polarity
        # Save user context
        st.session_state["conversation_history"].append(
            f"User choice: {user_choice}, Mindset: {mindset_text}, Sentiment: {sentiment_score}"
        )

        # 2. Universe splitter logic
        futures = predict_multiverse(user_choice, sentiment_score)

        st.subheader("Your Possible Futures:")
        for idx, (fut, prob) in enumerate(futures, start=1):
            # 3. GPT-based explanation
            explanation = generate_explanation(
                choice=user_choice,
                future=fut,
                user_history="\n".join(st.session_state["conversation_history"])
            )
            st.write(f"**{idx}.** {fut} (Probability: {round(prob * 100, 2)}%)")
            st.write(f"*Explanation:* {explanation}")
            st.write("---")

        # Show sentiment
        if sentiment_score > 0:
            mindset_label = "positive"
        elif sentiment_score < 0:
            mindset_label = "negative"
        else:
            mindset_label = "neutral"

        st.info(f"Sentiment Score: {round(sentiment_score, 2)} - Your mindset is **{mindset_label}**")

st.write("© 2025 Multiverse AI. For entertainment purposes only.")
