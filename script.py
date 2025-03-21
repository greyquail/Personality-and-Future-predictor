#!/usr/bin/env python
"""
Personality and Future Predictor App
======================================

This Streamlit app demonstrates a sample "advanced AI" application
that predicts possible futures based on a user’s decision and mindset,
and then generates an explanation for each future using a local causal
language model (GPT-Neo 2.7B from EleutherAI).

It uses the Hugging Face Transformers library to load a model and tokenizer,
and employs Streamlit caching to avoid reloading the model on every run.

The app is intended for experimental/educational purposes only.

Requirements:
  - streamlit>=1.22.0
  - torch
  - transformers
  - huggingface-hub
  - (other dependencies as listed in requirements.txt)

Instructions:
  1. Save this file as script.py in your GitHub repository.
  2. Push your changes to GitHub.
  3. Deploy on Streamlit Community Cloud or run locally via:
        streamlit run script.py
  4. Ensure that any secrets (if used) are managed via Streamlit secrets,
     but in this example we rely solely on local model inference.
     
Note: This app does not rely on external API calls so no secret tokens are required.
"""

import os
import random
import time
import logging
import warnings

# Suppress some warnings for clarity
warnings.filterwarnings("ignore", category=FutureWarning)

# Import Streamlit and dependencies
import streamlit as st

# For deep learning model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# For additional text processing and utility
from textblob import TextBlob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Global Constants and Configuration
# =============================================================================

MODEL_NAME = "EleutherAI/gpt-neo-2.7B"  # The model we use for prediction and explanation
CACHE_PERSIST = True  # Set persist=True to enable disk caching (if needed)
GENERATED_MAX_LENGTH = 150  # Maximum length for generated explanations
TEMPERATURE = 0.7         # Sampling temperature for generation

# =============================================================================
# Helper Functions
# =============================================================================

@st.cache_resource(persist=CACHE_PERSIST, show_spinner=False)
def load_model():
    """
    Load the GPT-Neo model and tokenizer from Hugging Face.

    Uses the st.cache_resource decorator to cache the model and tokenizer
    so that they are loaded only once per session.
    
    Returns:
        tokenizer (PreTrainedTokenizer): The loaded tokenizer.
        model (PreTrainedModel): The loaded GPT-Neo model.
    """
    try:
        st.write("Loading model and tokenizer (this may take a few minutes)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Load model with low CPU memory usage and float16 (adjust as needed)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        logger.info("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.error("Error loading model: %s", e)
        st.error("There was an error loading the model. Please try again later.")
        raise

def predict_possible_futures(choice: str) -> list:
    """
    Generate a list of possible future outcomes based on the user's input.
    
    This function simulates the process by randomly selecting from a predefined
    list of potential outcomes. In a more advanced app, you could incorporate
    sentiment analysis or other logic here.

    Args:
        choice (str): The user's decision or input.

    Returns:
        List[Tuple[str, int]]: A list of tuples, each containing a future prediction
                               and an assigned probability percentage.
    """
    base_futures = [
        "You will become a renowned entrepreneur with global recognition.",
        "Your passion and creativity will lead to groundbreaking innovations.",
        "Challenges will transform you into a resilient leader.",
        "Your endeavors will bring you financial success and personal fulfillment.",
        "You will pioneer new trends in your field and inspire many.",
        "Unexpected opportunities will arise that reshape your future."
    ]
    # Ensure we sample no more futures than available
    sample_size = min(3, len(base_futures))
    selected = random.sample(base_futures, sample_size)
    # Randomly assign a probability percentage for each future
    futures_with_probs = [(future, random.randint(40, 100)) for future in selected]
    return futures_with_probs

def generate_explanation(choice: str, future: str, user_history: str) -> str:
    """
    Generate a detailed explanation for why the predicted future might occur.
    
    This function builds a prompt that includes the user's choice, history, and
    the predicted future. It then uses the loaded model to generate an explanation.
    
    Args:
        choice (str): The user's decision or input.
        future (str): The predicted future outcome.
        user_history (str): A summary of the user's past decisions or mindset.

    Returns:
        str: The generated explanation text.
    """
    prompt = (
        f"User's choice: {choice}\n"
        f"User history: {user_history}\n"
        f"Predicted Future: {future}\n\n"
        "Explain in detail, with mystical yet logical reasoning, why this future "
        "might occur. Your explanation should be thoughtful, insightful, and "
        "provide a narrative that connects the user's past decisions to this outcome."
    )
    try:
        tokenizer, model = load_model()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # Generate text using sampling
        outputs = model.generate(
            **inputs,
            max_length=GENERATED_MAX_LENGTH,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation
    except Exception as e:
        logger.error("Error generating explanation: %s", e)
        return "Sorry, we could not generate an explanation at this time."

def analyze_user_input(text: str) -> dict:
    """
    Perform a simple analysis on the user's input.
    
    Here we use TextBlob for a basic sentiment analysis. (This is a placeholder
    for a more complex personality analysis.)
    
    Args:
        text (str): The user's text input.

    Returns:
        dict: Analysis results including sentiment polarity and subjectivity.
    """
    blob = TextBlob(text)
    sentiment = blob.sentiment
    analysis = {
        "polarity": sentiment.polarity,
        "subjectivity": sentiment.subjectivity
    }
    return analysis

def display_divider():
    """Utility function to display a horizontal divider."""
    st.write("---")

# =============================================================================
# Main App Function
# =============================================================================

def main():
    """
    Main function to run the Streamlit app.
    
    It sets up the sidebar for user inputs, displays the analysis,
    predicts possible future outcomes, and shows generated explanations.
    """
    st.set_page_config(page_title="Personality and Future Predictor",
                       layout="wide")
    
    # Display the title and description
    st.title("Personality and Future Predictor")
    st.markdown("""
    Welcome to the Personality and Future Predictor app. This experimental app 
    uses a deep learning model (GPT-Neo 2.7B) to generate possible future outcomes 
    based on your recent decisions and mindset. Enter your details in the sidebar 
    and click **Predict My Future**.
    """)
    
    # Sidebar inputs for user details
    st.sidebar.header("Your Input")
    user_choice = st.sidebar.text_input("Enter your recent decision or choice:", 
                                        value="I started a new business.")
    user_mindset = st.sidebar.radio("How do you feel about this decision?",
                                    options=["Optimistic", "Cautious", "Indifferent", "Anxious"])
    
    # Display user input on the main page
    st.write("### Your Decision Details")
    st.write(f"**Decision:** {user_choice}")
    st.write(f"**Mindset:** {user_mindset}")
    
    # Analyze the input using a simple sentiment analysis
    analysis = analyze_user_input(user_choice)
    st.write("**Input Analysis:**")
    st.write(f"Polarity: {analysis['polarity']:.2f}, Subjectivity: {analysis['subjectivity']:.2f}")
    
    display_divider()
    
    # Button to trigger prediction
    if st.button("Predict My Future"):
        with st.spinner("Generating predictions and explanations..."):
            # Generate a combined user history string
            user_history = f"Decision: {user_choice} | Mindset: {user_mindset}"
            
            # Predict possible futures
            futures = predict_possible_futures(user_choice)
            st.subheader("Your Possible Futures:")
            
            for idx, (fut, prob) in enumerate(futures, start=1):
                st.write(f"**Future {idx}:** {fut} _(Probability: {prob}% )_")
                explanation = generate_explanation(user_choice, fut, user_history)
                st.write(f"**Explanation:** {explanation}")
                display_divider()
    
    # Additional features: display last updated time and footer
    st.write("")
    st.write("App last updated at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    st.markdown("""
    ---
    **Disclaimer:** The predictions and explanations generated by this app are entirely
    based on a language model and are for entertainment and experimental purposes only.
    """)
    
    # Extra verbose logging to help debugging (prints to console)
    logger.info("User input processed: %s, Mindset: %s", user_choice, user_mindset)
    logger.info("Analysis: %s", analysis)

# =============================================================================
# Run the App
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.exception("Unhandled exception in the app:")
