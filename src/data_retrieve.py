from src.agent import get_agent, get_thread_id, logger

def main():
    """Main function to run the command-line chatbot interface."""

    # Get the agent loaded with the System Prompt
    agent = get_agent()

    # Get the current Thread id
    thread_id = get_thread_id()
    config = {"configurable": {"thread_id": thread_id}}

    print("Medical Chatbot Ready. Type 'exit' to quit.\n")

    # Run the chatbot.
    while True:
        try:
            print('\n')
            query = input("You: ").strip()
            
            if query.lower() == "exit":
                print("\nThank you for using Medical Chatbot.")
                print("Goodbye!")
                logger.info(f"Session {thread_id} Ended.")
                break
            
            if not query:
                print("Please enter a question.\n")
                continue

            # Using streaming mode for messages and retrieval
            for stream_mode, chunk in agent.stream({
                "messages": [{"role": "user", "content": query}]
            }, config=config, stream_mode=["messages", "custom"]):
                
                if stream_mode == "custom":
                    print(chunk)
                elif stream_mode == "messages":
                    token, metadata = chunk[0], chunk[1]
                    if metadata['langgraph_node'] == "model":
                        print(token.content, end='')

        except KeyboardInterrupt:
            logger.info(f"Session {thread_id} interrupted.")
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error in session {thread_id}: {str(e)}", exc_info=True)
            print(f"\nError: {str(e)}\n")

if __name__ == '__main__':
    main()

