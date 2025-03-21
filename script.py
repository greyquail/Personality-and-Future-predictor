import streamlit as st
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------
# Load AI Model (Open-Source Mistral-7B)
# --------------------------------------------------------------
def load_model():
    """Load Mistral-7B Instruct model (quantized for CPU)."""
    model_name = "TheBloke/Mistral-7B-Instruct-GGUF"  # Open-source, no authentication needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

# Load Model (No Cache to Avoid Issues)
tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --------------------------------------------------------------
# Generate Multiple Futures (Multiverse Simulation)
# --------------------------------------------------------------
def predict_multiverse(choice, sentiment_score):
    """
    Generate multiple possible futures with assigned probabilities.
    """
    possible_futures = [
        "You achieve extreme success and gain global recognition.",
        "You struggle but achieve moderate success.",
        "You face difficulties and learn from failures.",
        "You discover an unexpected opportunity leading to success.",
        "Your journey takes an entirely different path."
    ]
    
    # Generate probabilities based on sentiment analysis
    if sentiment_score > 0.2:
        num_choices = 3
    elif sentiment_score < -0.2:
        num_choices = 2
    else:
        num_choices = 4

    selected_futures = random.sample(possible_futures, num_choices)
    return selected_futures

# --------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------
st.title("🔮 AI-Powered Personality & Future Predictor")
st.write("Discover your potential future based on your personality and choices!")

# User Input
user_input = st.text_area("Enter your personality traits or current situation:")

if st.button("Predict Future"):
    if user_input:
        with st.spinner("Analyzing your personality..."):
            # Encode input and generate response
            inputs = tokenizer(user_input, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=100)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)

            # Simulate sentiment score (random for now)
            sentiment_score = random.uniform(-1, 1)
            
            # Predict multiple possible futures
            futures = predict_multiverse(user_input, sentiment_score)

            st.subheader("🌟 Possible Futures:")
            for future in futures:
                st.write(f"- {future}")

            st.subheader("🔮 AI Insight:")
            st.write(prediction)
    else:
        st.warning("Please enter your personality traits or situation.")
