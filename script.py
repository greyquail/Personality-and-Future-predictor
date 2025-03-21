"""
Personality & Future Predictor
================================

This is a sample Streamlit app that loads the GPT‑Neo 2.7B model and uses it
to generate possible future outcomes and an explanation for those outcomes.
It uses caching to keep the model and tokenizer in memory to speed up
predictions.

Requirements:
  - streamlit
  - transformers
  - torch
  - requests
  - textblob
  - nltk
  - accelerate

Ensure you have installed these packages in your environment.
"""

import os
import time
import random
import warnings
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from textblob import TextBlob

# Disable some warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set some global parameters
MODEL_NAME = "EleutherAI/gpt-neo-2.7B"
CACHE_ENABLED = True  # adjust if you wish to clear cache
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_OUTPUT_TOKENS = 150

# Define a constant for cache key (if needed)
CACHE_KEY_MODEL = "gpt_neo_model"
CACHE_KEY_TOKENIZER = "gpt_neo_tokenizer"

# -----------------------------------------------------------------------------
# Helper functions and caching decorators
# -----------------------------------------------------------------------------

# Note: st.cache_resource (introduced in later Streamlit versions) caches resource objects.
# Remove unsupported parameters (such as persist) if not accepted.
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer():
    """
    Load the GPT-Neo 2.7B model and tokenizer from Hugging Face.
    This function is cached so that subsequent runs do not reload the model.
    """
    st.write("Loading model and tokenizer... (this might take a few minutes)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to(MODEL_DEVICE)
    return tokenizer, model

def generate_text(prompt, model, tokenizer):
    """
    Generate text from a prompt using the loaded model.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(MODEL_DEVICE)
    output_ids = model.generate(
        input_ids,
        do_sample=True,
        max_new_tokens=MAX_OUTPUT_TOKENS,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=1
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

def predict_possible_futures(choice):
    """
    Dummy function that returns a list of possible future outcomes.
    In a real scenario, you could call a model or use complex logic.
    """
    base_futures = [
        "You will become an entrepreneur.",
        "You will have a stable career in technology.",
        "You will travel the world and learn diverse cultures.",
        "You will experience rapid personal growth.",
        "You will become a thought leader in your field."
    ]
    # For example, sample outcomes based on choice length
    if len(choice.split()) < 3:
        selected = random.sample(base_futures, 2)
    else:
        selected = random.sample(base_futures, 3)
    # Assign dummy probabilities (for demonstration purposes)
    futures_with_probs = [(future, round(random.uniform(0.6, 1.0), 2)) for future in selected]
    return futures_with_probs

def generate_explanation(choice, future, user_history):
    """
    Generate an explanation for why a particular future might occur.
    Uses the ChatCompletion API of a language model (here we use a dummy call).
    In this demo, we simulate the explanation.
    """
    # Create a detailed prompt for explanation.
    prompt = (
        f"User choice: {choice}\n"
        f"User history: {user_history}\n"
        f"Predicted future: {future}\n\n"
        "Explain in detail why this future might occur, considering the user's mindset and past decisions. "
        "Use a mystical yet logical tone and provide insights into the decisions leading to this outcome."
    )
    # Use our generate_text function to simulate explanation generation.
    explanation = generate_text(prompt, model, tokenizer)
    return explanation

def clear_cache():
    """
    Clear Streamlit's cache.
    """
    st.cache_resource.clear()
    st.cache_data.clear()
    st.success("Cache cleared.")

# -----------------------------------------------------------------------------
# Main application logic
# -----------------------------------------------------------------------------

def main():
    st.title("Personality and Future Predictor")
    st.write("Enter your current decision and mindset to predict possible futures.")
    
    # Sidebar for settings and cache clearing
    st.sidebar.header("Settings")
    if st.sidebar.button("Clear Cache"):
        clear_cache()
    
    # Input fields
    user_choice = st.text_input("Enter your current decision:", placeholder="e.g., Starting my own business")
    user_mindset = st.text_input("Enter your current mindset:", placeholder="e.g., Optimistic and curious")
    
    # Display conversation history (simulate a session memory)
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []
    
    if st.button("Predict My Future"):
        if user_choice and user_mindset:
            # Append user input to conversation history
            st.session_state["conversation_history"].append(f"Choice: {user_choice} | Mindset: {user_mindset}")
            # Load model and tokenizer (cached)
            global tokenizer, model
            tokenizer, model = load_model_and_tokenizer()
            
            # Predict futures based on user choice
            futures = predict_possible_futures(user_choice)
            
            st.subheader("Your Possible Futures:")
            for idx, (fut, prob) in enumerate(futures, start=1):
                st.write(f"**Option {idx}:** {fut} (Confidence: {int(prob * 100)}%)")
                # Generate an explanation using our model
                explanation = generate_explanation(user_choice, fut, "\n".join(st.session_state["conversation_history"]))
                st.write("Explanation:")
                st.write(explanation)
                st.write("---")
        else:
            st.error("Please provide both your decision and mindset.")
    
    # Display the conversation history
    if st.checkbox("Show conversation history"):
        st.write("\n".join(st.session_state["conversation_history"]))

# -----------------------------------------------------------------------------
# Run the app
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
