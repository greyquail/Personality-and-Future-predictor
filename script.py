import streamlit as st
import random
from textblob import TextBlob
import warnings

# Optional: Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# 1) Advanced Model Setup: GPT-Neo 2.7B
# -----------------------------------------------------------------------------
# This model is more advanced than distilgpt2 but requires more resources.
model_name = "EleutherAI/gpt-neo-2.7B"

@st.cache_resource(show_spinner=False)
def load_model():
    # Use half precision if a GPU is available
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    return tokenizer, model

tokenizer, model = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_text(prompt, max_length=250, temperature=0.7, top_p=0.9, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
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
    # If the sentiment is somewhat positive, choose 3; otherwise, choose 2
    if sentiment_score > 0.1:
        chosen = random.sample(base_futures, 3)
    else:
        chosen = random.sample(base_futures, 2)
    
    results = []
    for fut in chosen:
        prob = random.uniform(0.3, 0.8)
        if sentiment_score > 0.1:
            prob += 0.1
        prob = min(prob, 0.9)  # cap probability to 0.9
        results.append((fut, round(prob, 2)))
    return results

def generate_explanation(choice, future, user_history):
    """
    Uses a chain-of-thought prompt to try and get a more logical explanation.
    Note: GPT-Neo-2.7B is advanced compared to smaller models, but its reasoning is still limited.
    """
    prompt = f"""
You are a wise AI that sees multiple timelines and reasons logically.
User Decision: {choice}
Predicted Future: {future}
Conversation History: 
{user_history}

Now, provide a detailed explanation using bullet points:
- Briefly analyze the decision.
- List two or three logical influences that could lead to the predicted future.
- Summarize with a concise, coherent explanation.

Do not simply repeat the user's input; provide fresh reasoning.
"""
    return generate_text(prompt, max_length=250)

# -----------------------------------------------------------------------------
# 3) Streamlit App UI & Session Setup
# -----------------------------------------------------------------------------
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

st.set_page_config(page_title="Advanced Future Predictor", layout="centered")
st.title("🔮 Advanced Future Predictor")
st.write("""
This app uses GPT-Neo 2.7B from Hugging Face to generate fictional predictions and explanations.
**Note**: This is for educational and demonstration purposes only.
""")

user_choice = st.text_input("Enter a significant decision or scenario")
mindset_text = st.text_area("Describe your current mindset or situation (be as detailed as possible)")

if st.button("Predict My Future"):
    if not user_choice or not mindset_text:
        st.warning("Please provide both a decision and your current mindset.")
    else:
        # Analyze sentiment using TextBlob
        sentiment_score = TextBlob(mindset_text).sentiment.polarity
        
        # Update conversation history
        new_entry = f"Decision: {user_choice}, Mindset: {mindset_text}, Sentiment: {sentiment_score}"
        st.session_state["conversation_history"].append(new_entry)
        
        # Generate possible futures
        futures = predict_multiverse(user_choice, sentiment_score)
        
        st.subheader("Your Possible Futures:")
        for idx, (fut, prob) in enumerate(futures, start=1):
            conv_str = "\n".join(st.session_state["conversation_history"])
            explanation = generate_explanation(user_choice, fut, conv_str)
            st.markdown(f"**{idx}.** {fut} (Probability: {prob*100:.1f}%)")
            st.markdown(f"**Explanation:** {explanation}")
            st.markdown("---")
        
        # Display sentiment info
        if sentiment_score > 0.1:
            label = "positive"
        elif sentiment_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        st.info(f"Sentiment Score: {round(sentiment_score, 2)} — Your mindset is {label}")

st.write("© 2025 Advanced Future Predictor. For educational purposes only.")
