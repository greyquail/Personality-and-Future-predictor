import streamlit as st
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from textblob import TextBlob

# --------------------------------------------------------------
# Load AI Model (GPT-Neo 2.7B) with Caching
# --------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    """Load GPT-Neo 2.7B model from Hugging Face."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-2.7B", force_download=True, resume_download=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-2.7B", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            force_download=True, 
            resume_download=False
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None

# Load Model
tokenizer, model = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
if model:
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

    results = []
    for fut in selected_futures:
        probability = round(random.uniform(0.3, 0.9), 2)  # Assign random probability
        results.append((fut, probability))

    return results

# --------------------------------------------------------------
# Explanation Generation (Why This Future?)
# --------------------------------------------------------------
def generate_explanation(choice, future):
    """
    Generate an explanation of why the predicted future might occur.
    """
    if not model:
        return "⚠️ Model not loaded. Unable to generate explanation."

    prompt = f"""
    You are an AI analyzing multiple parallel universes.
    The user made a decision: {choice}
    The predicted future: {future}
    
    Explain why this outcome is likely. Consider personality, global trends, and hidden factors.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_new_tokens=150, temperature=0.7)
    explanation = tokenizer.decode(output[0], skip_special_tokens=True)
    return explanation

# --------------------------------------------------------------
# Sentiment Analysis of User Input
# --------------------------------------------------------------
def get_sentiment(text):
    """
    Analyze sentiment polarity (-1 to +1) based on user input.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity

# --------------------------------------------------------------
# Streamlit UI Setup
# --------------------------------------------------------------
st.title("🔮 Multiverse Future Predictor AI")
st.write("Enter a decision and let AI predict multiple possible futures.")

# Sidebar Inputs
st.sidebar.header("User Input")
user_choice = st.sidebar.text_input("Enter a major decision:", placeholder="e.g., Start my own business")
user_mindset = st.sidebar.text_input("Describe your mindset:", placeholder="e.g., Optimistic and determined")

# Run Prediction
if st.sidebar.button("Predict My Future"):
    if not user_choice or not user_mindset:
        st.sidebar.warning("Please provide both a decision and your mindset.")
    else:
        st.sidebar.info("Analyzing possible timelines...")
        
        # Get Sentiment Score
        sentiment_score = get_sentiment(user_mindset)
        
        # Generate Multiple Futures
        predictions = predict_multiverse(user_choice, sentiment_score)
        
        # Display Results
        st.subheader("Possible Futures")
        for idx, (future, prob) in enumerate(predictions, start=1):
            explanation = generate_explanation(user_choice, future)
            st.write(f"**{idx}. {future} (Probability: {prob*100:.1f}%)**")
            st.write(f"📜 **Explanation:** {explanation}")
            st.write("---")

# Display Debugging Information
st.sidebar.subheader("Debug Info")
st.sidebar.write(f"Sentiment Score: {round(sentiment_score, 2)}")
st.sidebar.write(f"Device in Use: {device}")

st.write("🔍 AI analyzes decision-based parallel universes to provide precise predictions.")
