import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import warnings

# Suppress certain FutureWarnings from libraries
warnings.filterwarnings("ignore", category=FutureWarning)

@st.cache_data(show_spinner=False)
def load_model():
    """
    Loads GPT-Neo 2.7B from Hugging Face once and caches it.
    Make sure your environment has 'transformers' and 'torch' installed.
    Also pinned 'numpy<2.0' in requirements.txt to avoid version conflicts.
    """
    model_name = "EleutherAI/gpt-neo-2.7B"
    # Set dtype to float16 if memory is tight. Use float32 if issues arise.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

def predict_possible_futures(user_choice):
    """
    A toy function that picks random 'futures' for demonstration.
    Returns a list of (future_description, probability).
    """
    # For example, 3 random 'futures':
    FUTURES = [
        "faced enormous challenges",
        "reinvented your entire career",
        "achieved massive success",
        "discovered an unexpected path",
    ]
    # Randomly pick 2 or 3
    picks = random.sample(FUTURES, k=2)
    # Assign random probabilities
    results = []
    for p in picks:
        prob = round(random.uniform(40.0, 80.0), 1)  # random 40%–80%
        results.append((p, prob))
    return results

def generate_explanation(user_choice, future, user_history, tokenizer, model):
    """
    Calls GPT-Neo to produce a short explanation of WHY that future might occur.
    We craft a prompt with a 'chain-of-thought' style or mystical style.
    """
    prompt = f"""You are a mystical, wise AI analyzing multiple timelines.
The user made a decision: {user_choice}
We predicted the user might: {future}

Conversation / user history: {user_history}

Reasoning (chain-of-thought):
  1) Consider the user's choice and mindset.
  2) List possible influences or reasons that could lead to "{future}".
  3) Summarize logically in bullet points.

Final Explanation:
Write a coherent explanation of WHY this future might happen, in a mystical yet logical style. 
Include bullet points from the reasoning. Avoid repeating user input verbatim too many times.
Use your intuition to help the user understand.
---
"""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")
    # You may want to set max_new_tokens or other generate() params
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # (Optional) clean up repeated prompt
    # A simple approach is to remove everything before the last "Final Explanation:"
    splitted = explanation.split("Final Explanation:")
    if len(splitted) > 1:
        explanation = "Final Explanation:" + splitted[-1]
    return explanation.strip()

def main():
    st.title("Multiverse Future Predictor")
    st.write("A demonstration of using GPT-Neo (2.7B) in Streamlit for hypothetical future predictions.")

    # Let user input a choice or scenario
    user_choice = st.text_input("Describe your decision or scenario (e.g., 'I moved from UK to India'):")
    user_mindset = st.text_input("Describe your mindset (e.g., 'excited, created an AI in 2 days'):")
    user_history = f"Choice: {user_choice}, Mindset: {user_mindset}"

    if st.button("Predict My Future"):
        tokenizer, model = load_model()
        # 1) Predict possible futures
        futures = predict_possible_futures(user_choice)

        st.subheader("Your Possible Futures:")
        for idx, (fut, prob) in enumerate(futures, start=1):
            st.markdown(f"**{idx}.** {fut} *(Probability: {prob}%)*")
            # 2) Generate explanation from GPT-Neo
            explanation = generate_explanation(user_choice, fut, user_history, tokenizer, model)
            st.markdown(f"**Explanation:** {explanation}")
            st.write("---")

if __name__ == "__main__":
    main()
