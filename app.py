import streamlit as st
import logging
from datetime import datetime
from src.agent import get_agent
from rate_limit import RateLimit
import uuid

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medbot.log'),
    ]
)

logger = logging.getLogger(__name__)

# Log app startup
logger.info("=" * 50)
logger.info("MedBot application started")
logger.info("=" * 50)

# ============================================
# HELPER FUNCTIONS
# ============================================



def get_user_ip():
    try:
        ip = st.context.ip_address
        logger.debug(f"Retrieved IP address: {ip}")
        return ip
    except Exception as e:
        logger.error(f"Failed to get IP address: {e}", exc_info=True)
        return "unknown"

# ============================================
# STREAMLIT UI SETUP
# ============================================
st.set_page_config(page_title="MedBot", page_icon=":robot_face:")
st.title("MedBot: Your Medical Chatbot Assistant")

rate_limiter = RateLimit()

# Initialize session state for chat history and thread_id
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.info("Initialized new chat session")

if "thread_id" not in st.session_state:
    st.session_state.thread_id =  str(uuid.uuid4())
    logger.info(f"Created new thread_id: {st.session_state.thread_id}")

if "agent" not in st.session_state:
    st.session_state.agent = get_agent()
    logger.info(f"Created the agent and stored in session")


# Disclaimer
with st.expander("Medical Disclaimer", expanded=False):
    st.warning(
        "This chatbot is created as a portfolio project and is for informational purposes only and is not a substitute "
        "for professional medical advice, diagnosis, or treatment. Always seek the "
        "advice of your physician or other qualified health provider."
    )

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================
# MAIN CHAT LOGIC
# ============================================
if prompt := st.chat_input("Ask a medical question"):
    user_ip = get_user_ip()
    
    # Log the incoming query
    logger.info(f"New query from IP {user_ip}: '{prompt[:100]}...'")  # Log first 100 chars
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # ============================================
    # RATE LIMITING CHECK
    # ============================================
    try:
        is_allowed, wait_time = rate_limiter.is_allowed(user_ip)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded. Wait time: {wait_time}s")
            st.error(
                f"⚠️ **Rate limit exceeded!**\n\n"
                f"You've reached the maximum of 5 requests per 30 minutes. "
                f"Please wait {wait_time} seconds before sending another message.\n\n"
                f"This limit helps us manage costs and ensure fair usage for all users."
            )
            st.stop()
        else:
            logger.info(f"Rate limit check passed ")
    
    except Exception as e:
        logger.error(f"Rate limiting error for IP : {e}", exc_info=True)
        st.error("An error occurred while checking rate limits. Please try again.")
        st.stop()

    # ============================================
    # GENERATE RESPONSE
    # ============================================
    with st.chat_message("assistant"):
        update_placeholder = st.empty()
        message_placeholder = st.empty()
        full_updates = ""
        full_response = ""
        
        try:
            start_time = datetime.now()
            logger.info(f"Starting agent processing for IP {user_ip}")
            
            agent = st.session_state.agent
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Stream response
            token_count = 0
            for stream_mode, chunk in agent.stream({
                "messages": [{"role": "user", "content": prompt}]
            }, config=config, stream_mode=["messages", "custom"]):

                if stream_mode == "custom":
                    # Handle custom streaming data (e.g., tool output)
                    logger.debug(f"Custom stream: {chunk}")
                    update_placeholder.markdown(f"*{chunk}*")
                    full_updates += chunk + "\n"
                    
                elif stream_mode == "messages":
                    # Handle message stream
                    token, metadata = chunk[0], chunk[1]
                    if metadata['langgraph_node'] == "model":
                        full_response += token.content
                        token_count += len(token.content.split())
                    message_placeholder.markdown(full_response + "▌")
            
            # Final response
            message_placeholder.markdown(full_response)
            
            # Calculate response time
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Log success
            logger.info(
                f"Response generated successfully for IP {user_ip} | "
                f"Response time: {response_time:.2f}s | "
                f"Response length: {len(full_response)} chars | "
                f"Estimated tokens: {token_count}"
            )
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        except Exception as e:
            # Log the error with full traceback
            logger.error(
                f"Error generating response for IP {user_ip}: {e}",
                exc_info=True  # This logs the full stack trace
            )
            
            # Show user-friendly error
            st.error(
                "Sorry, something went wrong while generating the response. "
                "Our team has been notified. Please try again in a moment."
            )
  

logger.info("Chat interaction completed")