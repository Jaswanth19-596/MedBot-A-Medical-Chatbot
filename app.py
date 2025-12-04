import streamlit as st
from src.agent import get_agent, get_thread_id

st.set_page_config(page_title="MedBot", page_icon=":robot_face:")

st.title("MedBot: Your Medical Chatbot Assistant")

# Initialize session state for chat history and thread_id
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = get_thread_id()

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
