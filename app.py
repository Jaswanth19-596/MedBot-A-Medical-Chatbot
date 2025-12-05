import streamlit as st
from src.agent import get_agent, get_thread_id
from rate_limit import RateLimit

def get_user_ip():
    try:
        ip = st.context.ip_address
        return ip
    except Exception as e:
        return "unknown"

st.set_page_config(page_title="MedBot", page_icon=":robot_face:")

st.title("MedBot: Your Medical Chatbot Assistant")

rate_limiter = RateLimit()

# Initialize session state for chat history and thread_id
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = get_thread_id()

# Disclaimer
with st.expander("Medical Disclaimer", expanded=False):
    st.warning(
        "This chatbot is for informational purposes only and is not a substitute "
        "for professional medical advice, diagnosis, or treatment. Always seek the "
        "advice of your physician or other qualified health provider."
    )


# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a medical question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)


    # USE THE SESSION STATE RATE LIMITER
    is_allowed, wait_time = rate_limiter.is_allowed(get_user_ip())

    if not is_allowed:
        st.error(
            f"⚠️ **Rate limit exceeded!**\n\n"
            f"You've reached the maximum of 10 requests per minute. "
            f"Please wait {wait_time} seconds before sending another message.\n\n"
            f"This limit helps us manage costs and ensure fair usage for all users."
        )
        st.stop()

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        agent = get_agent()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        for stream_mode, chunk in agent.stream({
            "messages": [{"role": "user", "content": prompt}]
        }, config=config, stream_mode=["messages", "custom"]):

            if stream_mode == "custom":
                # Handle custom streaming data (e.g., tool output)
                message_placeholder.markdown(f"*{chunk}*")
            elif stream_mode == "messages":
                # Handle message stream
                token, metadata = chunk[0], chunk[1]
                if metadata['langgraph_node'] == "model":
                        full_response += token.content
                # full_response += token.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
