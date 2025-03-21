import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from textblob import TextBlob
import warnings

# Suppress FutureWarnings (optional)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# 1) Advanced Model Setup: GPT-Neo 2.7B using Accelerate
# -----------------------------------------------------------------------------
model_name = "EleutherAI/gpt-neo-2.7B"

def load_model():
    """
    Loads GPT-Neo 2.7B from Hugging Face with low_cpu_mem_usage.
    Note: Using low_cpu_mem_usage requires the accelerate package.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

# Load model each time (removing caching to avoid local disk cache issues)
tokenizer, model = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_text(prompt, max_length=250, temperature=0.7, top_p=0.9, top_k=50):
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
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------------------------------------------------------
# 2) Future Prediction Logic ("Multiverse")
# -----------------------------------------------------------------------------
def predict_multiverse(user_choice, sentiment_score):
    base_futures = [
        "achieved massive success",
        "discovered an unexpected path",
        "faced enormous challenges",
        "changed the world in a surprising way",
        "reinvented your entire career"
    ]
    # Choose 3 futures if sentiment is positive, otherwise 2
    if sentiment_score > 0.1:
        chosen = random.sample(base_futures, 3)
    else:
        chosen = random.sample(base_futures, 2)
    
    results = []
    for fut in chosen:
        prob = random.uniform(0.3, 0.8)
        if sentiment_score > 0.1:
            prob += 0.1
        prob = min(prob, 0.9)
        results.append((fut, round(prob, 2)))
    return results

def generate_explanation(choice, future, user_history):
    prompt = f"""
You are a mystical, wise AI analyzing multiple timelines.
User Decision: {choice}
Predicted Future: {future}

Conversation History:
{user_history}

REASONING (chain-of-thought):
- Analyze the user's decision and mindset.
- List two to three logical factors that might influence this future.
- Summarize the overall outcome in clear bullet points.

FINAL EXPLANATION:
Provide a coherent, mystical yet logical explanation of why this future might occur.
"""
    return generate_text(prompt, max_length=250)

# -----------------------------------------------------------------------------
# 3) Streamlit App UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Future Predictor", layout="centered")
st.title("🔮 Advanced Future Predictor")
st.write("""
This app uses GPT-Neo 2.7B from Hugging Face to generate fictional future predictions and explanations.
**Note:** This is a demonstration project for educational purposes.
""")

user_choice = st.text_input("Enter a significant decision or scenario:")
user_mindset = st.text_input("Describe your current mindset or situation:")
user_history = f"Decision: {user_choice}, Mindset: {user_mindset}"

if st.button("Predict My Future"):
    if not user_choice or not user_mindset:
        st.warning("Please provide both a decision and your current mindset.")
    else:
        # Analyze sentiment using TextBlob
        sentiment_score = TextBlob(user_mindset).sentiment.polarity
        # Generate possible futures
        futures = predict_multiverse(user_choice, sentiment_score)
        
        st.subheader("Your Possible Futures:")
        for idx, (fut, prob) in enumerate(futures, start=1):
            explanation = generate_explanation(user_choice, fut, user_history)
            st.markdown(f"**{idx}.** {fut} (Probability: {prob*100:.1f}%)")
            st.markdown(f"**Explanation:** {explanation}")
            st.markdown("---")
        
        if sentiment_score > 0.1:
            label = "positive"
        elif sentiment_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        st.info(f"Sentiment Score: {round(sentiment_score, 2)} — Your mindset is {label}")

st.write("© 2025 Advanced Future Predictor. For educational purposes only.")
