import streamlit as st
import random
from textblob import TextBlob
import warnings

# Optional: Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# 1) Model Setup
# ---------------------------
model_name = "distilgpt2"

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_text(prompt, max_length=200, temperature=0.7, top_p=0.9, top_k=50):
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

# ---------------------------
# 2) Universe Splitter Logic
# ---------------------------
def predict_multiverse(user_choice, sentiment_score):
    base_futures = [
        "achieved massive success",
        "discovered an unexpected path",
        "faced enormous challenges",
        "changed the world in a surprising way",
        "reinvented your entire career"
    ]
    # If sentiment is somewhat positive, pick 3; else 2
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
    """
    Attempt a chain-of-thought style approach: ask for bullet points with reasons,
    then a short summary. The model might not always comply, given its small size.
    """
    prompt = f"""
You are a mystical, wise AI analyzing multiple timelines. 
The user made a decision: {choice}
We predicted the user might: {future}

Conversation / user history: 
{user_history}

### Reasoning (chain-of-thought)
- Step 1: Consider the user's choice and mindset.
- Step 2: List possible influences or reasons that could lead to "{future}".
- Step 3: Summarize logically in bullet points.

### Final Explanation
Write a coherent explanation of WHY this future might happen, using a mystical yet logical style. 
Include the bullet points from the reasoning. Avoid repeating user input verbatim too many times.
"""
    return generate_text(prompt, max_length=220)

# ---------------------------
# 3) Streamlit App
# ---------------------------
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

st.set_page_config(page_title="Logical Future Predictor", layout="centered")

st.title("🔮 Logical Future Predictor")
st.write("""
This app self-hosts a small model (distilgpt2). 
**Note**: This is just generative text, not a real future predictor.
""")

user_choice = st.text_input("Enter a significant decision or scenario")
mindset_text = st.text_area("Describe your mindset or situation briefly")

# Basic check for minimal length
if user_choice and len(user_choice.strip()) < 5:
    st.warning("Your decision/scenario is quite short. Provide more detail for better output.")
if mindset_text and len(mindset_text.strip()) < 5:
    st.warning("Your mindset text is very short. Provide more detail for better output.")

if st.button("Predict My Future"):
    if not user_choice or not mindset_text:
        st.warning("Please fill in both fields.")
    else:
        # Sentiment
        sentiment_score = TextBlob(mindset_text).sentiment.polarity
        
        # Update conversation history
        new_entry = f"Choice: {user_choice}, Mindset: {mindset_text}, Sentiment: {sentiment_score}"
        st.session_state["conversation_history"].append(new_entry)
        
        # Generate futures
        futures = predict_multiverse(user_choice, sentiment_score)
        
        st.subheader("Your Possible Futures:")
        for i, (fut, prob) in enumerate(futures, start=1):
            conv_str = "\n".join(st.session_state["conversation_history"])
            explanation = generate_explanation(user_choice, fut, conv_str)
            
            st.markdown(f"**{i}.** {fut} (Probability: {prob*100:.1f}%)")
            st.markdown(f"**Explanation:** {explanation}")
            st.markdown("---")
        
        # Display sentiment
        if sentiment_score > 0.1:
            label = "positive"
        elif sentiment_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        st.info(f"Sentiment Score: {round(sentiment_score, 2)} → Your mindset is {label}")

st.write("© 2025 Logical Future Predictor. For demonstration only.")
