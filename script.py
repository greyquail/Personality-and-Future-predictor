import streamlit as st
import requests
import random
from textblob import TextBlob

# -----------------------------------------------------------------------------
# 1) Hugging Face API Setup (via Streamlit Secrets)
# -----------------------------------------------------------------------------
# We retrieve the token from st.secrets. This avoids committing it in code.
HF_TOKEN = st.secrets["HF_TOKEN"]  # <-- Must match the key in your secrets toml
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Example model endpoint: GPT-Neo-2.7B
# You can also try EleutherAI/gpt-j-6B if you prefer a bigger model
HF_API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"

def query_huggingface(prompt):
    """
    Calls the Hugging Face Inference API with the given prompt.
    Returns the generated text or an error message.
    """
    payload = {"inputs": prompt}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # raise an error if not 2xx
        data = response.json()
        # Usually returns a list of dicts with 'generated_text'
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "")
        else:
            return str(data)  # fallback if format is unexpected
    except requests.exceptions.RequestException as e:
        return f"Error calling Hugging Face API: {e}"

# -----------------------------------------------------------------------------
# 2) Universe Splitter Logic (example)
# -----------------------------------------------------------------------------
def predict_multiverse(user_choice, sentiment_score):
    """
    Returns a list of (future_variant, probability).
    We'll adjust probabilities by sentiment.
    """
    base_futures = [
        "achieved massive success",
        "discovered an unexpected path",
        "faced enormous challenges",
        "changed the world in a surprising way"
    ]
    # For demonstration, pick 3 if sentiment is positive, else 2
    if sentiment_score > 0:
        chosen_futures = random.sample(base_futures, 3)
    else:
        chosen_futures = random.sample(base_futures, 2)
    
    futures_with_probs = []
    for fut in chosen_futures:
        prob = random.uniform(0.3, 0.8)
        # Shift probabilities if sentiment is positive
        if sentiment_score > 0:
            prob += 0.1
        prob = round(prob, 2)
        futures_with_probs.append((fut, prob))
    
    return futures_with_probs

# -----------------------------------------------------------------------------
# 3) Explanation or "Why" Generator
# -----------------------------------------------------------------------------
def generate_explanation(choice, future, user_history):
    """
    Uses the Hugging Face Inference API to generate an explanation
    for why this future might occur, referencing user context.
    """
    prompt = f"""
You are a wise AI that can see multiple timelines.

User's choice: {choice}
Predicted future: {future}
User's conversation/history: {user_history}

Explain in detail WHY this future might occur, considering the user's mindset.
Use a mystical yet logical style.
"""
    return query_huggingface(prompt)

# -----------------------------------------------------------------------------
# 4) Streamlit Session Setup
# -----------------------------------------------------------------------------
# We store conversation or user data in session_state
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# -----------------------------------------------------------------------------
# 5) Streamlit UI
# -----------------------------------------------------------------------------
st.title("🔮 Advanced Future Predictor (Hugging Face Edition)")
st.write("""
This app uses a free Hugging Face model (GPT-Neo-2.7B) to generate text.
It also demonstrates a "multiverse" future approach and sentiment-based logic.
""")

user_choice = st.text_input("Enter a past decision or question")
mindset_text = st.text_area("Describe your current mindset or situation")

if st.button("Predict My Future"):
    if not user_choice or not mindset_text:
        st.warning("Please provide both your decision and mindset.")
    else:
        # 1) Analyze sentiment
        sentiment_score = TextBlob(mindset_text).sentiment.polarity
        
        # 2) Append to conversation history
        st.session_state["conversation_history"].append(
            f"User choice: {user_choice}, Mindset: {mindset_text}, Sentiment: {sentiment_score}"
        )
        
        # 3) Predict multiple futures
        futures = predict_multiverse(user_choice, sentiment_score)
        
        st.subheader("Your Possible Futures:")
        for idx, (fut, prob) in enumerate(futures, start=1):
            # 4) Generate an explanation for each future
            explanation = generate_explanation(
                choice=user_choice,
                future=fut,
                user_history="\n".join(st.session_state["conversation_history"])
            )
            st.write(f"**{idx}.** {fut} (Probability: {prob*100:.1f}%)")
            st.write(f"*Explanation:* {explanation}")
            st.write("---")
        
        # 5) Show sentiment info
        if sentiment_score > 0:
            mindset_label = "positive"
        elif sentiment_score < 0:
            mindset_label = "negative"
        else:
            mindset_label = "neutral"
        
        st.info(f"Sentiment Score: {round(sentiment_score, 2)} – Your mindset is **{mindset_label}**")

st.write("© 2025 Multiverse AI. For entertainment purposes only.")
