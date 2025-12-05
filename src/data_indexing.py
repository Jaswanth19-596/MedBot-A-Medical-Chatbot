from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from src.helpers import create_vectorstore, load_config
from src.data_ingestion import load_and_filter_documents
from src.data_chunking import split_text_into_chunks
from dotenv import load_dotenv


def main():
    load_dotenv()

    # Loading the configuration values of the model
    config = load_config()

    # Variables
    embedding_model = config['embeddings']['name']
    dimensions = config['embeddings']['dimensions']
    data_path = config['paths']['data_path']

    # Load and filter the documents from the data folder
    documents = load_and_filter_documents(data_path)

    # Split the documents into chunks
    chunks = split_text_into_chunks(documents)

    # Creating the object of the Embeddings model
    embedding_model = OpenAIEmbeddings(model = embedding_model, dimensions= dimensions)

    # # Set the name of the index
    index_name = config['index_name']

    # # Create the vector store
    create_vectorstore(index_name=index_name)

    # # Get the index
    pc = Pinecone()
    index = pc.Index(index_name)

    # # Pass the index and embedding_model to the Pinecone Vector store
    vector_store = PineconeVectorStore(index = index, embedding = embedding_model)

    # # Add the chunks to the vector store.
    vector_store.add_documents(chunks)


if __name__ == '__main__':
    main()