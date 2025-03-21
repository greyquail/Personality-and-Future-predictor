# =============================================================================
# Advanced Future Predictor App
#
# This is a Streamlit application that loads GPT-Neo 2.7B from Hugging Face
# using low_cpu_mem_usage (which requires the accelerate package). It then
# simulates "future predictions" based on a user’s decision and mindset.
#
# The app uses internal functions and a simple conversation manager that
# saves history to a local JSON file. It does not rely on external caching,
# so it should work in environments like Streamlit Cloud.
#
# Note: GPT-Neo 2.7B is a huge model and may be slow or memory‑intensive in
# free deployments. Consider using a smaller model if needed.
# =============================================================================

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from textblob import TextBlob
import json
import os
import datetime
import warnings
import requests  # needed for potential future extension
import logging

# -----------------------------------------------------------------------------
# Logging and Warning Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Global Constants and File Paths
# -----------------------------------------------------------------------------
MODEL_NAME = "EleutherAI/gpt-neo-2.7B"
CONVERSATION_CACHE_FILE = "conversation_cache.json"

# -----------------------------------------------------------------------------
# Utility Functions for Time and History Management
# -----------------------------------------------------------------------------
def get_current_timestamp():
    """Return the current timestamp as a formatted string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_conversation_history(history):
    """Save conversation history to a JSON file."""
    try:
        with open(CONVERSATION_CACHE_FILE, "w") as f:
            json.dump(history, f)
    except Exception as e:
        logger.error(f"Error saving history: {e}")

def load_conversation_history():
    """Load conversation history from a JSON file."""
    if os.path.exists(CONVERSATION_CACHE_FILE):
        try:
            with open(CONVERSATION_CACHE_FILE, "r") as f:
                history = json.load(f)
            return history
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return []
    return []

def append_to_history(new_entry):
    """Append a new entry to the conversation history and save."""
    history = load_conversation_history()
    history.append(new_entry)
    save_conversation_history(history)
    return history

# -----------------------------------------------------------------------------
# Model Loading Function with Caching in Memory
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached():
    """
    Loads GPT-Neo 2.7B with low_cpu_mem_usage enabled.
    Uses the accelerate package for efficient memory use.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

# Try to load the model
try:
    tokenizer, model = load_model_cached()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------------------------------------------------------
# Text Generation Function
# -----------------------------------------------------------------------------
def generate_text(prompt, max_length=250, temperature=0.7, top_p=0.9, top_k=50):
    """Generate text from a prompt using the loaded model."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    except Exception as e:
        logger.error(f"Text generation error: {e}")
        return "Error generating text."

# -----------------------------------------------------------------------------
# Sentiment Analysis
# -----------------------------------------------------------------------------
def get_sentiment(text):
    """Return the sentiment polarity of the input text."""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return 0.0

# -----------------------------------------------------------------------------
# Future Prediction Logic ("Multiverse")
# -----------------------------------------------------------------------------
def predict_multiverse(user_choice, sentiment_score):
    """
    Predict a set of possible future outcomes based on the user decision and sentiment.
    Returns a list of tuples: (future description, probability).
    """
    base_futures = [
        "achieved massive success",
        "discovered an unexpected path",
        "faced enormous challenges",
        "changed the world in a surprising way",
        "reinvented your entire career",
        "found a hidden passion",
        "became a thought leader in your field"
    ]
    try:
        if sentiment_score > 0.1:
            num_choices = 3
        else:
            num_choices = 2
        chosen = random.sample(base_futures, num_choices)
    except Exception as e:
        logger.error(f"Error selecting future outcomes: {e}")
        chosen = random.sample(base_futures, 2)
    
    results = []
    for fut in chosen:
        prob = random.uniform(0.3, 0.8)
        if sentiment_score > 0.1:
            prob += 0.1
        prob = min(prob, 0.9)
        results.append((fut, round(prob, 2)))
    return results

# -----------------------------------------------------------------------------
# Explanation Generation
# -----------------------------------------------------------------------------
def generate_explanation(choice, future, user_history):
    """
    Generate a detailed explanation of why the predicted future might occur.
    """
    prompt = f"""
You are a mystical, wise AI analyzing multiple timelines.
Timestamp: {get_current_timestamp()}
User Decision: {choice}
Predicted Future: {future}

Conversation History:
{user_history}

REASONING (chain-of-thought):
1. Analyze the context of the decision.
2. Consider the user's mindset and potential influences.
3. List two to three logical factors leading to this outcome.
4. Summarize the overall outcome in bullet points.

FINAL EXPLANATION:
Provide a clear, mystical yet logical explanation of why this future might occur.
"""
    explanation = generate_text(prompt, max_length=250)
    return explanation

# -----------------------------------------------------------------------------
# Conversation Manager Class
# -----------------------------------------------------------------------------
class ConversationManager:
    """
    Manages conversation history by loading, appending, and displaying entries.
    """
    def __init__(self):
        self.history = load_conversation_history()

    def add_entry(self, decision, mindset, prediction, explanation):
        entry = {
            "timestamp": get_current_timestamp(),
            "decision": decision,
            "mindset": mindset,
            "prediction": prediction,
            "explanation": explanation
        }
        self.history.append(entry)
        save_conversation_history(self.history)
        return self.history

    def get_history_text(self):
        if not self.history:
            return "No conversation history available."
        text_entries = []
        for idx, entry in enumerate(self.history, start=1):
            text_entries.append(
                f"{idx}. [{entry['timestamp']}]\n"
                f"Decision: {entry['decision']}\n"
                f"Mindset: {entry['mindset']}\n"
                f"Prediction: {entry['prediction']}\n"
                f"Explanation: {entry['explanation']}\n"
            )
        return "\n".join(text_entries)

conv_manager = ConversationManager()

# -----------------------------------------------------------------------------
# Additional Helper Functions
# -----------------------------------------------------------------------------
def initialize_session_state():
    """
    Initialize session state variables for user inputs and conversation.
    """
    if "user_choice" not in st.session_state:
        st.session_state["user_choice"] = ""
    if "user_mindset" not in st.session_state:
        st.session_state["user_mindset"] = ""
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = conv_manager.get_history_text()

def clear_cache():
    """
    Clear conversation cache from both session state and disk.
    """
    if os.path.exists(CONVERSATION_CACHE_FILE):
        os.remove(CONVERSATION_CACHE_FILE)
    st.session_state["conversation_history"] = ""
    conv_manager.history = []
    st.success("Cache cleared successfully.")

# -----------------------------------------------------------------------------
# Main App Function
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Advanced Future Predictor", layout="centered")
    st.title("🔮 Advanced Future Predictor")
    st.write("""
    This app uses GPT-Neo 2.7B from Hugging Face to generate fictional future predictions and detailed explanations.
    **Note:** This is a demonstration project for educational purposes.
    """)
    
    initialize_session_state()
    
    # Sidebar for inputs and cache clearing
    st.sidebar.header("User Input")
    user_choice = st.sidebar.text_input("Enter a significant decision or scenario:")
    user_mindset = st.sidebar.text_input("Describe your current mindset or situation:")
    
    st.sidebar.header("Actions")
    if st.sidebar.button("Clear Cache"):
        clear_cache()
    
    # Main button to predict future
    if st.sidebar.button("Predict My Future"):
        if not user_choice or not user_mindset:
            st.sidebar.warning("Please provide both a decision and your current mindset.")
            return
        
        # Analyze sentiment
        sentiment_score = get_sentiment(user_mindset)
        sentiment_label = "positive" if sentiment_score > 0.1 else ("negative" if sentiment_score < -0.1 else "neutral")
        st.sidebar.info(f"Sentiment Score: {round(sentiment_score, 2)} ({sentiment_label})")
        
        # Generate predictions
        predictions = predict_multiverse(user_choice, sentiment_score)
        
        st.subheader("Your Possible Futures:")
        for idx, (future, prob) in enumerate(predictions, start=1):
            explanation = generate_explanation(user_choice, future, f"Decision: {user_choice}, Mindset: {user_mindset}")
            st.markdown(f"**{idx}. {future} (Probability: {prob*100:.1f}%)**")
            st.markdown(f"**Explanation:** {explanation}")
            st.markdown("---")
            # Save the conversation entry
            conv_manager.add_entry(user_choice, user_mindset, future, explanation)
        
        # Update conversation history in session state
        st.session_state["conversation_history"] = conv_manager.get_history_text()
    
    # Display conversation history
    st.subheader("Conversation History")
    st.text_area("History", st.session_state.get("conversation_history", ""), height=300)
    
    # Additional Debug Info
    st.markdown("---")
    st.markdown("**Debug Info:**")
    st.markdown(f"- Device in use: **{device}**")
    st.markdown(f"- Model: **{MODEL_NAME}**")
    st.markdown(f"- Total history entries: **{len(conv_manager.history)}**")
    
    st.write("© 2025 Advanced Future Predictor. For educational purposes only.")
    st.write("This app relies on internal processing and local caching to minimize external calls.")

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        st.error(f"An unexpected error occurred: {err}")
        logger.exception("Unexpected error:")
