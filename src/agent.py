from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from dotenv import load_dotenv
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import Literal
from src.helpers import load_config

# ============= CONFIGURATION =============
load_dotenv()
config = load_config()

model = config['model']['name']
temperature = config['model']['temperature']
embedding_model_name = config['embeddings']['name']
dimensions = config['embeddings']['dimensions']
index_name = config['index_name']
search_type = config['retrieval']['search_type']
k = config['retrieval']['k']

# Logger Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler('app.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ============= INITIALIZATION =============

def get_retriever():
    """Initializes and returns the vector store retriever."""
    embedding_model = OpenAIEmbeddings(model=embedding_model_name, dimensions=dimensions)
    pc = Pinecone()
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
    return vector_store.as_retriever(search_type= search_type, search_kwargs={"k": k})

retriever = get_retriever()

# ============= TOOLS =============
class RelevanceOutput(BaseModel):
    output: Literal["yes", "no"]

def validate_relevance(query: str, docs: list) -> list:
    """Filter docs by relevance score"""
    relevance_checker_model = ChatOpenAI(model= model, temperature=temperature).with_structured_output(RelevanceOutput)
    
    filtered_docs = []
    for doc in docs:
        prompt = f"""Rate if this document is relevant to the query (yes/no) only:
            Query : {query}
            Document : {doc.page_content}
            Answer : 
        """
        response = relevance_checker_model.invoke(prompt)
        if response.output == 'yes':
            filtered_docs.append(doc)
    
    return filtered_docs


def rewrite_query(query: str) -> str:
    """The Main goal of this function is to rewrite the user's query to make it more searchable"""
    try:
        
        # Rewriting the user's query using LLM.
        rewriter = ChatOpenAI(model=model, temperature=temperature)

        # Prompt 
        rewrite_query = f"""Rewrite this medical question to be more specific and searchable.
            Keep it short (1-2) sentences.
            Original : {query}
            Rewritten : 
        """

        # Executing the LLM.
        rewritten = rewriter.invoke(rewrite_query).content

        return rewritten

    except Exception as e:
        logger.error("Error while Rewritting the Query", str(e))
        raise Exception



@tool
def retrieve_context(query: str):
    """Retrieve information to help answer a query"""
    try:
        # This writer is useful to give updates to the user 
        writer = get_stream_writer()

        rewritten_query = rewrite_query(query)
        
        # Retreiving the relevant documents from the vector store.
        retrieved_docs = retriever.invoke(rewritten_query)

        # If there are no relevant docs, just return empty
        if not retrieved_docs:
            writer('No relevant documents found.')
            logger.warning(f"No docs found for query: {rewritten_query}")
            return "No relevant information found.", []
        writer(f'Found {len(retrieved_docs)} sources. Checking for relevance...')

        # Validate the relevancy of the retrieved documents.
        filtered_docs = validate_relevance(query, retrieved_docs)

        # If none of them are relevant, return empty
        if not filtered_docs:
            writer('No relevant documents found after filtering.')
            return "No relevant information found.", []

        writer(f'Found {len(filtered_docs)} relevant sources.')

        # Join all the documents and return
        serialized = "\n\n".join((f"Source: {doc.metadata['book_name']} (Page: {doc.metadata['page']})\nContent: {doc.page_content}") for doc in filtered_docs)
        return serialized, filtered_docs
    
    except Exception as e:
        logger.error(f"Retrieval Error: {str(e)}", exc_info=True)
        writer(f'Error during retrieval: {str(e)}')
        return f"Error while retrieving documents: {str(e)}", []


# ============= AGENT SETUP =============
def get_agent():
    """Creates and returns the LangGraph agent."""
    system_prompt = """You are an expert Medical Chatbot assistant. Your role is to:

    1. Help users with medical questions using the medical database available to you.
    2. Use the retrieve_context tool to search the medical book for relevant information.
    3. Always cite the source of your information, including the book name and page number.
    4. If you don't find relevant information, say so clearly.
    5. Never make up medical information - only use retrieved context.
    6. Be empathetic and clear in your explanations.

    Important: This is NOT a replacement for professional medical advice. Always recommend consulting a healthcare provider for diagnosis or treatment decisions."""

    agent = create_agent(
        model=ChatOpenAI(model=model),
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver()
    )
    return agent

def get_thread_id():
    """Generates a unique thread ID for the chat session."""
    return f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
