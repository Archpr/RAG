import streamlit as st
import sys
import os
import base64

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ragchain.rag_chain import run_chain


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Kerala University Admission Assistant",
    page_icon="static/logo.png",
    layout="wide",
)

# ------------------ FUNCTIONS ------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ------------------ SIDEBAR ------------------
with st.sidebar:
    if os.path.exists("static/logo.png"):
        st.image("static/logo.png", width=150)
    
    st.title("Options")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat", type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.subheader("📜 Session History")
    if "messages" in st.session_state and st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                # Truncate long messages for the sidebar
                display_text = (msg['content'][:40] + '...') if len(msg['content']) > 40 else msg['content']
                st.caption(f"👤 {display_text}")
    else:
        st.caption("No history yet.")

# ------------------ BACKGROUND & CSS ------------------
# Adds your logo as the background (covers whole page)
if os.path.exists("static/logo.png"):
    bin_str = get_base64_of_bin_file("static/logo.png")
    background_style = f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: 20%; /* Watermark style */
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: center;
            }}

            /* Slight white overlay for readability */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(255, 255, 255, 0.93);
                z-index: -1;
            }}
        </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Center Chat Input */
        .stChatInput {
            position: fixed;
            bottom: 2rem;
            left: 50%;
            transform: translateX(-50%);
            width: 70%;
            max-width: 800px;
            z-index: 1000;
        }
        
        div[data-testid="stChatInput"] {
             margin: 0 auto;
        }

        /* Adjust container padding to prevent content from being hidden behind fixed input */
        .block-container {
            padding-bottom: 120px;
            padding-top: 2rem;
        }
        
        /* Chat bubble styling */
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)


# ------------------ APP TITLE ------------------
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>Kerala University Admission Assistant </h1>",
                unsafe_allow_html=True)


# ------------------ CHAT HISTORY ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ------------------ USER INPUT ------------------
prompt = st.chat_input("Type your question here...")

if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get RAG Response
    with st.spinner("Thinking..."):
        try:
            response = run_chain(prompt)
        except Exception as e:
            response = f"Sorry, I encountered an error: {e}"

    # Display response
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update sidebar history immediately
    st.rerun()
