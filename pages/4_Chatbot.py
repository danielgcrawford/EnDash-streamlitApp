# pages/4_Chatbot.py
import streamlit as st
from lib import auth

st.set_page_config(page_title="Chatbot", page_icon="🧠", layout="centered")
auth.require_login()
auth.render_sidebar()

st.title("🧠 Chatbot (placeholder)")

st.info(
    "This page will embed a chatbot (Zapier Chatbots or Microsoft Copilot Studio) "
    "and scope it to the currently selected dataset."
)
