import streamlit as st
import random
from textblob import TextBlob
import warnings

# Suppress FutureWarnings (optional)
warnings.filterwarnings("ignore", category=FutureWarning)

# Force-load below-2.0 numpy if environment has something else
# (This is a safeguard; if pinned in requirements.txt, it should already do so.)
# import subprocess
# subprocess.run(["pip", "install", "numpy<2.0"])

# Transformers & Torch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# 1) Self-Hosted Model Setup
# -----------------------------------------------------------------------------
model_name = "distilgpt2"

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_text(prompt, max_length=120, temperature=0.7, top_p=0.9, top_k=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------------------------------------------------------
# 2) "Multiverse" Future Logic
# -----------------------------------------------------------------------------
def predict_multiverse(user_choice, sentiment_score):
    """
    Return a list of (future_variant, probability) based on user input & sentiment.
    This is purely for demonstration, not real prediction.
    """
    base_futures = [
        "achieved massive success",
        "discovered an unexpected path",
        "faced enormous challenges",
        "changed the world in a surprising way",
        "reinvented your entire career"
    ]
    # If sentiment is positive, pick 3 outcomes; else pick 2
    if sentiment_score > 0.1:
        chosen = random.sample(base_futures, 3)
    else:
        chosen = random.sample(base_futures, 2)
    
    results = []
    for fut in chosen:
        prob = random.uniform(0.3, 0.8)
        if sentiment_score > 0.1:
            prob += 0.1
        prob = min(prob, 0.9)  # cap at 90%
        results.append((fut, round(prob, 2)))
    return results

def generate_explanation(choice, future, user_history):
    """
    Produces a textual 'explanation' using the local distilgpt2 model.
    This is purely generative text, not a real future prediction.
    """
    # Create a more structured, longer prompt
    prompt = f"""You are a mystical AI with deep wisdom, analyzing multiple possible timelines.
The user made a decision: {choice}
We predicted the user might: {future}

Conversation / user history: 
{user_history}

Write a short but coherent explanation of WHY this future might happen, 
in a mystical yet logical style. 
Explain possible reasons, influences, or outcomes. 
Avoid repeating user input verbatim too many times.
"""
    return generate_text(prompt, max_length=180)

# -----------------------------------------------------------------------------
# 3) Streamlit App
# -----------------------------------------------------------------------------
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

st.set_page_config(page_title="Enhanced Future Predictor", layout="centered")

st.title("🔮 Enhanced Future Predictor")
st.write("""
This self-hosted app uses a small local model (distilgpt2) to generate 
fictional "futures" and explanations. No external tokens or APIs are used, 
so there are no usage limits from third-party services.

**Disclaimer**: This is for demonstration only, not actual future prediction!
""")

# Input fields
user_choice = st.text_input("Enter a significant decision or question")
mindset_text = st.text_area("Describe your mindset or situation in a sentence or two")

# Basic check for minimal length
if user_choice and len(user_choice.strip()) < 5:
    st.warning("Your decision input is very short. Please provide a bit more detail.")
if mindset_text and len(mindset_text.strip()) < 5:
    st.warning("Your mindset text is very short. Try writing a full sentence for better results.")

if st.button("Predict My Future"):
    if not user_choice or not mindset_text:
        st.warning("Please fill in both fields.")
    else:
        # Sentiment analysis
        sentiment_score = TextBlob(mindset_text).sentiment.polarity
        
        # Append to conversation history
        new_entry = f"Choice: {user_choice}, Mindset: {mindset_text}, Sentiment: {sentiment_score}"
        st.session_state["conversation_history"].append(new_entry)
        
        # Generate possible futures
        futures = predict_multiverse(user_choice, sentiment_score)
        
        st.subheader("Your Possible Futures:")
        for i, (fut, prob) in enumerate(futures, start=1):
            # Build conversation string
            conv_str = "\n".join(st.session_state["conversation_history"])
            explanation = generate_explanation(user_choice, fut, conv_str)
            
            st.write(f"**{i}.** {fut} (Probability: {prob*100:.1f}%)")
            st.write(f"**Explanation:** {explanation}")
            st.write("---")
        
        # Display sentiment
        if sentiment_score > 0.1:
            label = "positive"
        elif sentiment_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        st.info(f"Sentiment Score: {round(sentiment_score, 2)} → Your mindset is {label}")

st.write("© 2025 Enhanced Future Predictor. For demonstration only.")
