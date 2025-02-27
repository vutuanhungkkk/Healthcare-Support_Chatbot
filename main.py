import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

model = T5ForConditionalGeneration.from_pretrained("./chatbot_model")
tokenizer = T5Tokenizer.from_pretrained("./chatbot_model")
device = model.device

def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.strip().lower()
    return text

def chatbot(dialogue):
    dialogue = clean_text(dialogue)
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=250)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(
        inputs["input_ids"],
        max_length=250,
        num_beams=4,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Medical Chatbot", layout="centered")
st.title("Medical Chatbot")
st.write("Ask your health-related questions below:")
user_message = st.text_input("Your Message:")

if st.button("Send"):
    if user_message.strip() != "":
        st.session_state.chat_history.append(("You", user_message))
        with st.spinner("Generating response..."):
            bot_response = chatbot(user_message)
        st.session_state.chat_history.append(("Bot", bot_response))
    else:
        st.error("Please enter a message.")

for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
