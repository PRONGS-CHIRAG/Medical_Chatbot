import streamlit as st
from app.rag_pipeline import medical_chatbot

def app():
    st.set_page_config(page_title="🩺 Medical Chat Assistant", layout="wide")
    st.title("🩺 Medical Chat Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a medical question:", "")

    if st.button("Send") and user_input.strip():
        st.session_state.chat_history.append({"sender": "User", "text": user_input})
        with st.spinner("Thinking..."):
            response = medical_chatbot(user_input)
            st.session_state.chat_history.append({"sender": "Assistant", "text": response})

    # Display messages
    for msg in st.session_state.chat_history:
        if msg["sender"] == "User":
            st.markdown(f"**🧑‍⚕️ You:** {msg['text']}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg['text']}")

