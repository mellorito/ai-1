import streamlit as st
from openai import OpenAI, APIError

st.title("Kimi / Llama Chatbot (NVIDIA NIM)")

# Safely get the API key from Streamlit secrets
try:
    api_key = st.secrets["NVIDIA_API_KEY"]
except Exception:
    st.error("Please add 'NVIDIA_API_KEY' to your Streamlit Secrets.")
    st.stop()

# Initialize OpenAI Client pointing to NVIDIA NIM Base URL
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        formatted_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        try:
            # Using the working model ID from NVIDIA Build
            completion = client.chat.completions.create(
                model="nvidia/llama-3.3-nemotron-super-49b-v1",
                messages=formatted_messages,
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
            )

            response_text = completion.choices[0].message.content
            st.markdown(response_text)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except APIError as e:
            st.error(f"NVIDIA API Error ({e.status_code}): {e.message}")
        except Exception as e:
            st.error(f"Error: {e}")
