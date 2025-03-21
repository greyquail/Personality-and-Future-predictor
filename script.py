import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------------------------------------------------------
# 1) Model Setup: Self-hosting a model using Transformers
# -----------------------------------------------------------------------------
# For a continuously available, self-hosted model without external rate limits,
# we load a model directly rather than using Hugging Face Inference API.
# We use "distilgpt2" as it is lightweight and can run on CPU.
model_name = "distilgpt2"

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Use GPU if available, otherwise CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_text(prompt, max_length=150, temperature=0.7, top_p=0.95, top_k=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # Generate text using the model
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# -----------------------------------------------------------------------------
# 2) Advanced "Multiverse" Logic (Example)
# -----------------------------------------------------------------------------
def predict_multiverse(user_choice, sentiment_score):
    """
    Returns a list of (future_variant, probability) tuples based on user input and sentiment.
    This is a simple example logic.
    """
    base_futures = [
        "achieved massive success",
        "discovered an unexpected path",
        "faced enormous challenges",
        "changed the world in a surprising way"
    ]
    # If sentiment is positive, choose 3 outcomes; otherwise, choose 2.
    if sentiment_score > 0:
        chosen_futures = random.sample(base_futures, 3)
    else:
        chosen_futures = random.sample(base_futures, 2)
    
    futures_with_probs = []
    for fut in chosen_futures:
        prob = random.uniform(0.3, 0.8)
        if sentiment_score > 0:
            prob += 0.1
        futures_with_probs.append((fut, round(prob, 2)))
    
    return futures_with_probs

def generate_explanation(choice, future, user_history):
    """
    Generates an explanation for why a given future might occur using our self-hosted model.
    This concatenates the input into a prompt and uses generate_text().
    """
    prompt = f"""
You are a wise AI that sees multiple timelines.

User's choice: {choice}
Predicted future: {future}
User conversation/history: {user_history}

Explain in detail WHY this future might occur, using a mystical yet logical tone.
"""
    return generate_text(prompt, max_length=200)

# -----------------------------------------------------------------------------
# 3) Streamlit UI and Session Setup
# -----------------------------------------------------------------------------
import random
from textblob import TextBlob

# Initialize conversation history if not already present
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

st.set_page_config(page_title="Unlimited Future Predictor", layout="centered")
st.title("🔮 Unlimited Future Predictor")
st.write("""
This app is self-hosted using a free model (distilgpt2) from the Transformers library.
It avoids external usage limits by loading the model directly.
Enter a past decision and describe your mindset.
""")

user_choice = st.text_input("Enter a past decision or question")
mindset_text = st.text_area("Describe your current mindset or situation")

if st.button("Predict My Future"):
    if not user_choice or not mindset_text:
        st.warning("Please provide both your decision and your current mindset.")
    else:
        # Analyze sentiment using TextBlob
        sentiment_score = TextBlob(mindset_text).sentiment.polarity
        
        # Append the input to session conversation history
        st.session_state["conversation_history"].append(
            f"Choice: {user_choice}, Mindset: {mindset_text}, Sentiment: {sentiment_score}"
        )
        
        # Predict possible futures based on user input and sentiment
        futures = predict_multiverse(user_choice, sentiment_score)
        
        st.subheader("Your Possible Futures:")
        for idx, (fut, prob) in enumerate(futures, start=1):
            explanation = generate_explanation(
                choice=user_choice,
                future=fut,
                user_history="\n".join(st.session_state["conversation_history"])
            )
            st.write(f"**{idx}.** {fut} (Probability: {prob*100:.1f}%)")
            st.write(f"*Explanation:* {explanation}")
            st.write("---")
        
        # Display sentiment info
        if sentiment_score > 0:
            mindset_label = "positive"
        elif sentiment_score < 0:
            mindset_label = "negative"
        else:
            mindset_label = "neutral"
        st.info(f"Sentiment Score: {round(sentiment_score, 2)} — Your mindset is {mindset_label}")

st.write("© Preetham 2025 Unlimited Future Predictor. For educational purposes only.")
