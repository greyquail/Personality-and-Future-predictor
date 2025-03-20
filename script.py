import streamlit as st
import requests
import random
from textblob import TextBlob

# -----------------------------------------------------------------------------
# 1) Hugging Face Inference API Setup (Token from Secrets)
# -----------------------------------------------------------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]  # Must match the key in your secrets
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# The correct model name for GPT-Neo-2.7B
HF_API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"

def query_huggingface(prompt):
    """
    Calls the Hugging Face Inference API with the given prompt.
    Returns generated text or an error message.
    """
    payload = {"inputs": prompt}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise if 4xx/5xx
        data = response.json()
        # Usually returns a list of dicts with 'generated_text'
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "")
        else:
            return str(data)  # fallback if unexpected
    except requests.exceptions.RequestException as e:
        return f"Error calling Hugging Face API: {e}"

# -----------------------------------------------------------------------------
# 2) Universe Splitter Logic
# -----------------------------------------------------------------------------
def predict_multiverse(user_choice, sentiment_score):
    """
    Returns a list of (future_variant, probability).
    Adjust probabilities by sentiment.
    """
    base_futures = [
        "achieved massive success",
        "discovered an unexpected path",
        "faced enormous challenges",
        "changed the world in a surprising way"
    ]
    # Example: If sentiment>0, pick 3 futures; else 2
    if sentiment_score > 0:
        chosen_futures = random.sample(base_futures, 3)
    else:
        chosen_futures = random.sample(base_futures, 2)
    
    futures_with_probs = []
    for fut in chosen_futures:
        prob = random.uniform(0.3, 0.8)
        if sentiment_score > 0:
            prob += 0.1
        prob = round(prob, 2)
        futures_with_probs.append((fut, prob))
    
    return futures_with_probs

# -----------------------------------------------------------------------------
# 3) Explanation Function
# -----------------------------------------------------------------------------
def generate_explanation(choice, future, user_history):
    """
    Creates a 'why' explanation using GPT-Neo-2.7B via Hugging Face.
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
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# -----------------------------------------------------------------------------
# 5) Streamlit UI
# -----------------------------------------------------------------------------
st.title("🔮 Future Predictor with GPT-Neo-2.7B")
st.write("""
This app uses Hugging Face's free GPT-Neo-2.7B model to generate text.
We do advanced "multiverse" logic and explanation based on sentiment.
""")

user_choice = st.text_input("Enter a past decision or question")
mindset_text = st.text_area("Describe your current mindset or situation")

if st.button("Predict My Future"):
    if not user_choice or not mindset_text:
        st.warning("Please provide both your decision and mindset.")
    else:
        # Analyze sentiment
        sentiment_score = TextBlob(mindset_text).sentiment.polarity
        
        # Append to conversation history
        st.session_state["conversation_history"].append(
            f"User choice: {user_choice}, Mindset: {mindset_text}, Sentiment: {sentiment_score}"
        )
        
        # Predict multiple futures
        futures = predict_multiverse(user_choice, sentiment_score)
        
        st.subheader("Your Possible Futures:")
        for idx, (fut, prob) in enumerate(futures, start=1):
            # Generate explanation
            explanation = generate_explanation(
                choice=user_choice,
                future=fut,
                user_history="\n".join(st.session_state["conversation_history"])
            )
            st.write(f"**{idx}.** {fut} (Probability: {prob*100:.1f}%)")
            st.write(f"*Explanation:* {explanation}")
            st.write("---")
        
        # Show sentiment info
        if sentiment_score > 0:
            mindset_label = "positive"
        elif sentiment_score < 0:
            mindset_label = "negative"
        else:
            mindset_label = "neutral"
        
        st.info(f"Sentiment Score: {round(sentiment_score, 2)} – Your mindset is **{mindset_label}**")

st.write("© 2025 Multiverse AI. For entertainment purposes only.")
