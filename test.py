import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-small"  # Lightweight model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set the padding token to be the same as the EOS token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Load model and tokenizer (cached)
model, tokenizer = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def chatbot_response(user_input):
    # Append the EOS token to signal end-of-sequence
    prompt = user_input + tokenizer.eos_token
    # Tokenize input with padding and truncation
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # Move all tensors to the selected device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # Generate output, using pad_token_id for proper padding
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        pad_token_id=tokenizer.pad_token_id
    )
    # Decode generated tokens and remove special tokens
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("Chatbot Demo (DialoGPT-small)")
st.write("Talk to our lightweight chatbot!")

user_input = st.text_input("You:")
if user_input:
    with st.spinner("Generating response..."):
        response = chatbot_response(user_input)
    st.text_area("Chatbot:", value=response, height=200, disabled=True)
