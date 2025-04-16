import streamlit as st
from application.rag_pipeline import medical_chatbot
from application.vector_store import *

st.set_page_config(page_title="🩺 Medical Chat Assistant", layout="wide")
st.title("🩺 Medical Chat Assistant")
dynamic_vs_creation()
with open("faiss_index.zip", "rb") as f:
        st.download_button("Download Embeddings", f, "faiss_index.zip")

# Initialize chat history and message counter if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "message_counter" not in st.session_state:
    st.session_state.message_counter = 0

# Display all previous exchanges and their responses
for i, msg in enumerate(st.session_state.chat_history):
    if msg["sender"] == "User":
        st.markdown(f"**🧑‍⚕️ You:** {msg['text']}")
    else:
        st.markdown(f"**🤖 Assistant:** {msg['text']}")
        st.write("---")  # Add separator after each exchange

# Create a new input for the current message
current_key = f"user_input_{st.session_state.message_counter}"
user_input = st.text_input("Ask a medical question:", key=current_key)
send_button = st.button("Send", key=f"send_{st.session_state.message_counter}")

# Process the user input when the button is clicked
if send_button and user_input.strip():
    # Add user message to chat history
    st.session_state.chat_history.append({"sender": "User", "text": user_input})
    
    # Get response from chatbot
    with st.spinner("Thinking..."):
        response = medical_chatbot(user_input)
        st.session_state.chat_history.append({"sender": "Assistant", "text": response})
    
    # Increment the message counter to create a new input field on next render
    st.session_state.message_counter += 1
    
    # Rerun to update the UI
    st.rerun()