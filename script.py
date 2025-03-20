import streamlit as st
import requests
import random
from textblob import TextBlob

# -----------------------------------------------------------------------------
# 1) Hugging Face Inference API Setup (Token from Secrets)
# -----------------------------------------------------------------------------
# The token is securely stored in Streamlit Cloud Secrets.
HF_TOKEN = st.secrets["HF_TOKEN"]
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Use the model endpoint for GPT-Neo-2.7B (ensure you have accepted its license on Hugging Face)
HF_API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"

def query_huggingface(prompt):
    """
    Calls the Hugging Face Inference API with the given prompt and returns the generated text.
    """
    payload = {"inputs": prompt}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for 4xx/5xx responses
        data = response.json()
        # Typically returns a list of dictionaries with a 'generated_text' key
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "")
        else:
            return str(data)  # Fallback if response format is unexpected
    except requests.exceptions.RequestException as e:
        return f"Error calling Hugging Face API: {e}"

# -----------------------------------------------------------------------------
# 2) Universe Splitter Logic: Predict Multiple Futures
# -----------------------------------------------------------------------------
def predict_multiverse(user_choice, sentiment_score):
    """
    Returns a list of tuples (future_variant, probability) based on the user's choice and sentiment.
    """
    base_futures = [
        "achieved massive success",
        "discovered an unexpected path",
        "faced enormous challenges",
        "changed the world in a surprising way"
    ]
    # If sentiment is positive, pick 3 outcomes; else pick 2.
    if sentiment_score > 0:
        chosen_futures = random.sample(base_futures, 3)
    else:
        chosen_futures = random.sample(base_futures, 2)
    
    futures_with_probs = []
    for fut in chosen_futures:
        prob = random.uniform(0.3, 0.8)
        # Adjust probability slightly based on sentiment
        if sentiment_score > 0:
            prob += 0.1
        prob = round(prob, 2)
        futures_with_probs.append((fut, prob))
    
    return futures_with_probs

# -----------------------------------------------------------------------------
# 3) Explanation Generator: Why This Future?
# -----------------------------------------------------------------------------
def generate_explanation(choice, future, user_history):
    """
    Generates a detailed explanation for why a given future might occur,
    combining the user’s choice, predicted future, and conversation history.
    """
    prompt = f"""
You are a wise AI that can see multiple timelines.

User's choice: {choice}
Predicted future: {future}
User conversation/history: {user_history}

Explain in detail WHY this future might occur, considering the user's mindset.
Use a mystical yet logical tone.
"""
    return query_huggingface(prompt)

# -----------------------------------------------------------------------------
# 4) Session Setup
# -----------------------------------------------------------------------------
# Initialize conversation history in session state.
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# -----------------------------------------------------------------------------
# 5) Streamlit User Interface
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Future Predictor", layout="centered")
st.title("🔮 Advanced Future Predictor (Hugging Face Edition)")
st.write("""
Enter a past decision or question and describe your current mindset.
Our AI will analyze your sentiment, predict multiple possible futures, and provide detailed explanations.
""")

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
        
        # Predict possible futures
        futures = predict_multiverse(user_choice, sentiment_score)
        
        st.subheader("Your Possible Futures:")
        for idx, (fut, prob) in enumerate(futures, start=1):
            # Generate explanation for each future
            explanation = generate_explanation(
                choice=user_choice,
                future=fut,
                user_history="\n".join(st.session_state["conversation_history"])
            )
            st.write(f"**{idx}.** {fut} (Probability: {prob*100:.1f}%)")
            st.write(f"*Explanation:* {explanation}")
            st.write("---")
        
        # Show sentiment information
        if sentiment_score > 0:
            mindset_label = "positive"
        elif sentiment_score < 0:
            mindset_label = "negative"
        else:
            mindset_label = "neutral"
        
        st.info(f"Sentiment Score: {round(sentiment_score, 2)} — Your mindset is {mindset_label}")

st.write("© 2025 Multiverse AI. For entertainment purposes only.")
