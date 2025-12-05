# MedBot API Reference

## Table of Contents
1. [Configuration (`helpers.py`)](#1-configuration-helperspy)
2. [Data Ingestion (`data_ingestion.py`)](#2-data-ingestion-data_ingestionpy)
3. [Data Chunking (`data_chunking.py`)](#3-data-chunking-data_chunkingpy)
4. [Data Indexing (`data_indexing.py`)](#4-data-indexing-data_indexingpy)
5. [Data Retrieval (`data_retrieve.py`)](#5-data-retrieval-data_retrievepy)
6. [Agent (`agent.py`)](#6-agent-agentpy)

---

## 1. Configuration (`helpers.py`)

This module contains helper functions for loading configuration from YAML files and environment variables.

### `load_config(config_path: str = 'config/config.yaml') -> dict`

Loads the main YAML configuration file.

*   **Parameters**:
    *   `config_path` (str): Path to the YAML configuration file. Defaults to `'config/config.yaml'`.
*   **Returns**:
    *   `dict`: A dictionary containing the configuration settings.
*   **Usage**:
    ```python
    from src.helpers import load_config

    config = load_config()
    print(config['llm']['model_name'])
    # Output: 'gpt-4o-mini'
    ```

### `get_api_key(key_name: str) -> str`

Retrieves an API key from environment variables.

*   **Parameters**:
    *   `key_name` (str): The name of the environment variable (e.g., `'OPENAI_API_KEY'`).
*   **Returns**:
    *   `str`: The API key.
*   **Raises**:
    *   `ValueError`: If the environment variable is not set.
*   **Usage**:
    ```python
    from src.helpers import get_api_key

    try:
        openai_key = get_api_key('OPENAI_API_KEY')
    except ValueError as e:
        print(e)
    ```

## 2. Data Ingestion (`data_ingestion.py`)

This module is responsible for loading PDF documents from a directory.

### `load_documents(data_dir: str) -> list`

Loads all PDF files from a specified directory.

*   **Parameters**:
    *   `data_dir` (str): The path to the directory containing PDF files.
*   **Returns**:
    *   `list`: A list of LangChain `Document` objects, where each object represents a PDF.
*   **Usage**:
    ```python
    from src.data_ingestion import load_documents

    documents = load_documents('data/')
    print(f"Loaded {len(documents)} documents.")
    ```

## 3. Data Chunking (`data_chunking.py`)

This module splits the loaded documents into smaller text chunks.

### `chunk_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list`

Splits a list of documents into smaller chunks using a text splitter.

*   **Parameters**:
    *   `documents` (list): A list of LangChain `Document` objects.
    *   `chunk_size` (int): The maximum number of characters per chunk.
    *   `chunk_overlap` (int): The number of characters to overlap between consecutive chunks.
*   **Returns**:
    *   `list`: A list of smaller `Document` objects representing the chunks.
*   **Usage**:
    ```python
    from src.data_chunking import chunk_documents
    from src.data_ingestion import load_documents
    from src.helpers import load_config

    config = load_config()
    documents = load_documents('data/')
    chunks = chunk_documents(
        documents,
        chunk_size=config['text_splitter']['chunk_size'],
        chunk_overlap=config['text_splitter']['chunk_overlap']
    )
    print(f"Created {len(chunks)} chunks.")
    ```

## 4. Data Indexing (`data_indexing.py`)

This module handles the creation of vector embeddings and storing them in Pinecone.

### `create_embeddings(api_key: str, model_name: str) -> OpenAIEmbeddings`

Initializes the OpenAI embedding model.

*   **Parameters**:
    *   `api_key` (str): The OpenAI API key.
    *   `model_name` (str): The name of the embedding model (e.g., `'text-embedding-3-small'`).
*   **Returns**:
    *   `OpenAIEmbeddings`: A LangChain `OpenAIEmbeddings` instance.

### `index_chunks(chunks: list, embeddings: OpenAIEmbeddings, index_name: str, api_key: str)`

Creates or updates a Pinecone index with the document chunks.

*   **Parameters**:
    *   `chunks` (list): A list of `Document` chunks to be indexed.
    *   `embeddings` (OpenAIEmbeddings): The embedding model instance.
    *   `index_name` (str): The name of the target Pinecone index.
    *   `api_key` (str): The Pinecone API key.
*   **Returns**:
    *   `None`.
*   **Error Handling**:
    *   Logs an error if the connection to Pinecone fails or if indexing is unsuccessful.
*   **Usage**:
    ```python
    # (assuming chunks and embeddings are created)
    from src.data_indexing import index_chunks
    from src.helpers import load_config, get_api_key

    config = load_config()
    pinecone_api_key = get_api_key('PINECONE_API_KEY')

    index_chunks(
        chunks=my_chunks,
        embeddings=my_embeddings_model,
        index_name=config['pinecone']['index_name'],
        api_key=pinecone_api_key
    )
    ```

## 5. Data Retrieval (`data_retrieve.py`)

This module is responsible for retrieving relevant documents from the vector store.

### `get_retriever(embeddings: OpenAIEmbeddings, index_name: str, api_key: str, k: int) -> PineconeVectorStore.as_retriever`

Initializes a retriever from an existing Pinecone index.

*   **Parameters**:
    *   `embeddings` (OpenAIEmbeddings): The embedding model instance.
    *   `index_name` (str): The name of the Pinecone index.
    *   `api_key` (str): The Pinecone API key.
    *   `k` (int): The number of top documents to retrieve (`Top-K`).
*   **Returns**:
    *   A LangChain retriever object configured for similarity search with `k` results.
*   **Usage**:
    ```python
    from src.data_retrieve import get_retriever
    # (assuming embeddings model is initialized)

    retriever = get_retriever(
        embeddings=my_embeddings_model,
        index_name='medbot',
        api_key='your-pinecone-key',
        k=3
    )
    ```

## 6. Agent (`agent.py`)

This module constructs and executes the main RAG chain.

### `create_rag_chain(retriever, llm, prompt_template: str) -> Runnable`

Creates the full RAG chain that connects the retriever, prompt, and LLM.

*   **Parameters**:
    *   `retriever`: The configured LangChain retriever from `get_retriever`.
    *   `llm`: The initialized Large Language Model (e.g., `ChatOpenAI`).
    *   `prompt_template` (str): A format string for the prompt, which must include `{context}` and `{question}` placeholders.
*   **Returns**:
    *   A LangChain `Runnable` object representing the entire RAG pipeline.
*   **Usage**:
    ```python
    from langchain_openai import ChatOpenAI
    from src.agent import create_rag_chain
    # (assuming retriever is initialized)

    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
    prompt = "Context: {context}\nQuestion: {question}\nAnswer:"
    
    rag_chain = create_rag_chain(retriever, llm, prompt)
    ```

### `get_response(query: str, chain: Runnable) -> str`

Invokes the RAG chain with a user query to get a final answer.

*   **Parameters**:
    *   `query` (str): The user's question.
    *   `chain` (Runnable): The compiled RAG chain.
*   **Returns**:
    *   `str`: The generated answer from the LLM.
*   **Error Handling**:
    *   This function may raise exceptions if the underlying API calls to the LLM fail (e.g., due to rate limits or network issues). It is recommended to wrap calls to this function in a `try...except` block.
*   **Usage**:
    ```python
    # (assuming rag_chain is created)
    from src.agent import get_response

    try:
        question = "What are the symptoms of diabetes?"
        response = get_response(question, rag_chain)
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")
    ```
