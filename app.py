# app.py
import streamlit as st
from llm import process_user_input  # atau core_agent.py

st.set_page_config(page_title="Chatbot Keuangan", page_icon="💬")
st.title("💬 Chatbot Keuangan")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ketik pesan...")

if user_input:
    st.chat_message("user").write(user_input)
    response = process_user_input(user_input)
    st.chat_message("assistant").write(response)

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
