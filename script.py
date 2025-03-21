#!/usr/bin/env python
# =============================================================================
# Advanced Future Predictor App
#
# This Streamlit app loads the GPT-Neo 2.7B model from Hugging Face and
# simulates future predictions based on a user's decision and mindset.
# It uses internal functions, a conversation manager, and extensive logging
# and error handling so that if an error occurs (for example, in model loading
# or text generation), the error is caught and displayed.
#
# Note: GPT-Neo 2.7B is extremely memory-intensive. For free deployments,
# consider switching to a smaller model (e.g., GPT-Neo 1.3B) if you encounter
# resource errors.
# =============================================================================

# -----------------------------------------------------------------------------
# Standard Imports and Logging Setup
# -----------------------------------------------------------------------------
import streamlit as st
import torch
import random
import os
import json
import datetime
import logging
import warnings

from textblob import TextBlob

# Import transformers components
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    st.error("Transformers library import failed. Please check your requirements.")
    raise e

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Global Constants and Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "EleutherAI/gpt-neo-2.7B"
# If needed, you can change MODEL_NAME to a smaller model for debugging.
# MODEL_NAME = "EleutherAI/gpt-neo-1.3B"

# File to store conversation history (if using local file caching)
CONVERSATION_FILE = "conversation_history.json"

# -----------------------------------------------------------------------------
# Helper Functions for Time, History, and File Handling
# -----------------------------------------------------------------------------
def get_timestamp():
    """Return the current timestamp as a string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_history(history):
    """Save the conversation history to a JSON file."""
    try:
        with open(CONVERSATION_FILE, "w") as f:
            json.dump(history, f)
    except Exception as err:
        logger.error(f"Failed to save history: {err}")

def load_history():
    """Load conversation history from a JSON file."""
    if os.path.exists(CONVERSATION_FILE):
        try:
            with open(CONVERSATION_FILE, "r") as f:
                history = json.load(f)
            return history
        except Exception as err:
            logger.error(f"Failed to load history: {err}")
            return []
    return []

def update_history(entry):
    """Append a new entry to the conversation history and save it."""
    history = load_history()
    history.append(entry)
    save_history(history)
    return history

# -----------------------------------------------------------------------------
# Model Loading Function (with in-memory caching)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model_resource():
    """
    Load the tokenizer and model using the specified model name.
    Uses low_cpu_mem_usage for efficiency.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        logger.info("Model loaded successfully.")
        return tokenizer, model
    except Exception as err:
        logger.exception("Error loading model:")
        raise err

# Load model (if error, display on UI)
try:
    tokenizer, model = load_model_resource()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------------------------------------------------------
# Text Generation Function
# -----------------------------------------------------------------------------
def generate_text(prompt, max_tokens=200, temperature=0.7, top_p=0.9, top_k=50):
    """Generate text from the prompt using the loaded model."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text
    except Exception as err:
        logger.exception("Text generation error:")
        return "Error generating response."

# -----------------------------------------------------------------------------
# Sentiment Analysis Function
# -----------------------------------------------------------------------------
def analyze_sentiment(text):
    """Return the sentiment polarity of the text (range -1 to 1)."""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception as err:
        logger.error(f"Sentiment analysis error: {err}")
        return 0.0

# -----------------------------------------------------------------------------
# Future Prediction (Multiverse) Function
# -----------------------------------------------------------------------------
def predict_futures(decision, sentiment):
    """
    Based on the user decision and sentiment, select possible future outcomes.
    Returns a list of (future, probability) tuples.
    """
    outcomes = [
        "achieved groundbreaking success",
        "faced unexpected setbacks",
        "discovered a new passion",
        "redefined your career",
        "became a thought leader",
        "experienced transformative change",
        "overcame major obstacles",
        "found inspiration in adversity"
    ]
    try:
        num_outcomes = 3 if sentiment > 0.05 else 2
        selected = random.sample(outcomes, num_outcomes)
    except Exception as err:
        logger.error(f"Error sampling outcomes: {err}")
        selected = outcomes[:2]
    # Assign random probabilities based on sentiment
    futures = []
    for outcome in selected:
        base_prob = random.uniform(0.4, 0.8)
        adjusted_prob = min(base_prob + (0.1 if sentiment > 0.05 else 0), 0.9)
        futures.append((outcome, round(adjusted_prob, 2)))
    return futures

# -----------------------------------------------------------------------------
# Explanation Generation Function
# -----------------------------------------------------------------------------
def explain_future(decision, future, history_text):
    """
    Generate a detailed explanation for why a given future might occur.
    Uses the generate_text function with a custom prompt.
    """
    prompt = f"""
You are an insightful, mystical AI tasked with explaining why a particular future outcome might occur.
Timestamp: {get_timestamp()}
User Decision: {decision}
Predicted Future: {future}

Conversation History:
{history_text}

Provide a detailed explanation with logical reasoning and mystical insight. List at least three factors that support this outcome and summarize them at the end.
"""
    explanation = generate_text(prompt, max_tokens=250)
    return explanation

# -----------------------------------------------------------------------------
# Conversation Manager Class
# -----------------------------------------------------------------------------
class ConversationManager:
    """Manages conversation history and formats it for display."""
    def __init__(self):
        self.history = load_history()

    def add_entry(self, decision, mindset, future, explanation):
        entry = {
            "timestamp": get_timestamp(),
            "decision": decision,
            "mindset": mindset,
            "predicted_future": future,
            "explanation": explanation
        }
        self.history.append(entry)
        save_history(self.history)
        return self.history

    def formatted_history(self):
        if not self.history:
            return "No history available."
        formatted = []
        for i, entry in enumerate(self.history, start=1):
            formatted.append(
                f"### Entry {i}\n"
                f"**Time:** {entry['timestamp']}\n\n"
                f"**Decision:** {entry['decision']}\n\n"
                f"**Mindset:** {entry['mindset']}\n\n"
                f"**Predicted Future:** {entry['predicted_future']}\n\n"
                f"**Explanation:** {entry['explanation']}\n\n---\n"
            )
        return "\n".join(formatted)

conv_manager = ConversationManager()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session():
    """Ensure that session state variables are initialized."""
    if "user_decision" not in st.session_state:
        st.session_state["user_decision"] = ""
    if "user_mindset" not in st.session_state:
        st.session_state["user_mindset"] = ""
    if "history_text" not in st.session_state:
        st.session_state["history_text"] = conv_manager.formatted_history()

init_session()

# -----------------------------------------------------------------------------
# Cache Clearing Function (for conversation history)
# -----------------------------------------------------------------------------
def clear_conversation():
    """Clear the conversation history both from the file and session state."""
    if os.path.exists(CONVERSATION_FILE):
        os.remove(CONVERSATION_FILE)
    conv_manager.history = []
    st.session_state["history_text"] = ""
    st.success("Conversation history cleared.")

# -----------------------------------------------------------------------------
# Main App Function
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Advanced Future Predictor", layout="centered")
    st.title("🔮 Advanced Future Predictor")
    st.markdown("""
    Welcome to the Advanced Future Predictor! Enter a significant decision and describe your current mindset.
    The app will analyze your input, predict possible future outcomes, and provide detailed explanations.
    **Note:** This project is for educational and demonstration purposes.
    """)
    
    # Sidebar for user inputs and cache management
    st.sidebar.header("Input & Actions")
    decision = st.sidebar.text_input("Enter your decision/scenario:")
    mindset = st.sidebar.text_input("Describe your current mindset:")
    
    if st.sidebar.button("Clear Conversation History"):
        clear_conversation()
    
    # Main button to trigger prediction
    if st.sidebar.button("Predict My Future"):
        if not decision or not mindset:
            st.sidebar.warning("Please provide both a decision and your mindset.")
        else:
            st.sidebar.info("Analyzing your input...")
            sentiment = analyze_sentiment(mindset)
            st.sidebar.write(f"Sentiment Score: {sentiment:.2f}")
            
            predictions = predict_futures(decision, sentiment)
            st.subheader("Possible Futures")
            for idx, (fut, prob) in enumerate(predictions, start=1):
                # For each predicted future, generate an explanation.
                explanation = explain_future(decision, fut, f"Decision: {decision}\nMindset: {mindset}")
                st.markdown(f"**{idx}. {fut} (Probability: {prob*100:.1f}%)**")
                st.markdown(f"*Explanation:* {explanation}")
                st.markdown("---")
                # Save each prediction in the conversation history.
                conv_manager.add_entry(decision, mindset, fut, explanation)
            
            # Update session state history display.
            st.session_state["history_text"] = conv_manager.formatted_history()

    # Display conversation history
    st.subheader("Conversation History")
    st.text_area("History", st.session_state.get("history_text", ""), height=300)
    
    # Debug information
    st.markdown("---")
    st.markdown("**Debug Info**")
    st.markdown(f"- **Device:** {device}")
    st.markdown(f"- **Model:** {MODEL_NAME}")
    st.markdown(f"- **Total History Entries:** {len(conv_manager.history)}")
    st.markdown(f"- **Timestamp:** {get_timestamp()}")
    
    st.markdown("---")
    st.markdown("© 2025 Advanced Future Predictor • Educational Project")

# -----------------------------------------------------------------------------
# App Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        st.error(f"An unexpected error occurred: {ex}")
        logger.exception("Unexpected error:")
