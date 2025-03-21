import streamlit as st
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------
# Load AI Model (Falcon-7B-Instruct Public Version)
# --------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Using a fully public model that does not require authentication.
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

# Load the model
tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --------------------------------------------------------------
# Generate Multiple Futures (Multiverse Simulation)
# --------------------------------------------------------------
def predict_multiverse(choice, sentiment_score):
    possible_futures = [
        "You achieve extreme success and gain global recognition.",
        "You struggle but achieve moderate success.",
        "You face difficulties and learn from failures.",
        "You discover an unexpected opportunity leading to success.",
        "Your journey takes an entirely different path."
    ]
    if sentiment_score > 0.2:
        num_choices = 3
    elif sentiment_score < -0.2:
        num_choices = 2
    else:
        num_choices = 4
    return random.sample(possible_futures, num_choices)

# --------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------
st.title("🔮 AI-Powered Personality & Future Predictor")
st.write("Discover your potential future based on your personality and choices!")

user_input = st.text_area("Enter your personality traits or current situation:")

if st.button("Predict Future"):
    if user_input:
        with st.spinner("Analyzing your personality..."):
            inputs = tokenizer(user_input, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=100)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Simulate a sentiment score (random for demonstration)
            sentiment_score = random.uniform(-1, 1)
            futures = predict_multiverse(user_input, sentiment_score)

            st.subheader("🌟 Possible Futures:")
            for future in futures:
                st.write(f"- {future}")

            st.subheader("🔮 AI Insight:")
            st.write(prediction)
    else:
        st.warning("Please enter your personality traits or situation.")
