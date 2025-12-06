# MedBot API Reference

This document provides a detailed reference for the functions and classes used in the MedBot application.

## Table of Contents
1. [Agent (`src/agent.py`)](#1-agent-srcagentpy)
2. [Data Indexing (`src/data_indexing.py`)](#2-data-indexing-srcdataindexingpy)
3. [Data Ingestion (`src/data_ingestion.py`)](#3-data-ingestion-srcdata_ingestionpy)
4. [Data Chunking (`src/data_chunking.py`)](#4-data-chunking-srcdatachunkingpy)
5. [Data Retrieval / CLI (`src/data_retrieve.py`)](#5-data-retrieval--cli-srcdataretrievepy)
6. [Helpers (`src/helpers.py`)](#6-helpers-srchelperspy)

---

## 1. Agent (`src/agent.py`)

This module is the core of the chatbot, defining the LangGraph agent and its entire workflow.

### `get_agent() -> Runnable`

Constructs and returns the complete, runnable LangGraph agent. This is the main entry point for interacting with the chatbot logic.

*   **Returns**:
    *   `Runnable`: A compiled LangGraph runnable that can be invoked with a user query.
*   **Usage**:
    ```python
    from src.agent import get_agent

    agent = get_agent()
    response = agent.invoke({"messages": [("user", "What is diabetes?")]})
    ```

### `create_agent(llm, retriever) -> Runnable`

Creates and compiles the LangGraph agent with its defined states, nodes, and edges.

*   **Parameters**:
    *   `llm`: An initialized language model (e.g., `ChatOpenAI`).
    *   `retriever`: An initialized vector store retriever.
*   **Returns**:
    *   `Runnable`: The compiled LangGraph agent.

### Agent Nodes

The agent's workflow is defined by a series of nodes:

*   **`rewrite_query(state)`**: Takes the user's query and uses the LLM to rewrite it for better search performance.
*   **`retrieve_context(state)`**: Uses the rewritten query to retrieve relevant document chunks from the Pinecone vector store.
*   **`validate_relevance(state)`**: Examines the retrieved documents and filters out any that are not relevant to the original user query.
*   **`generate_response(state)`**: Takes the filtered, relevant documents and generates a final answer using the LLM, grounded in the provided context.
*   **`fallback(state)`**: If no relevant documents are found, this node provides a helpful message to the user.

---

## 2. Data Indexing (`src/data_indexing.py`)

This script orchestrates the entire data preparation and indexing pipeline.

### `main()`

When run as a script, this function executes the full data pipeline:
1.  Loads configuration from `config/config.yaml`.
2.  Loads PDF documents from the `data/` directory.
3.  Splits the documents into smaller chunks.
4.  Initializes the OpenAI embedding model.
5.  Creates a Pinecone index if it doesn't exist.
6.  Adds the document chunks (as vector embeddings) to the Pinecone index.

*   **Usage**:
    ```bash
    python src/data_indexing.py
    ```

---

## 3. Data Ingestion (`src/data_ingestion.py`)

This module is responsible for loading documents from the file system.

### `load_and_filter_documents(data_path: str) -> list`

Loads all PDF documents from a specified directory, extracts their text, and cleans up metadata.

*   **Parameters**:
    *   `data_path` (str): The path to the directory containing the PDF files.
*   **Returns**:
    *   `list`: A list of LangChain `Document` objects.
*   **Usage**:
    ```python
    from src.data_ingestion import load_and_filter_documents

    documents = load_and_filter_documents("./data")
    ```

---

## 4. Data Chunking (`src/data_chunking.py`)

This module splits the loaded documents into smaller, more manageable chunks.

### `split_text_into_chunks(documents: list) -> list`

Takes a list of documents and splits them into smaller chunks based on the settings in `config/config.yaml`.

*   **Parameters**:
    *   `documents` (list): A list of LangChain `Document` objects.
*   **Returns**:
    *   `list`: A list of smaller `Document` objects (chunks).
*   **Usage**:
    ```python
    from src.data_chunking import split_text_into_chunks

    # Assuming 'documents' is a list of loaded documents
    chunks = split_text_into_chunks(documents)
    ```

---

## 5. Data Retrieval / CLI (`src/data_retrieve.py`)

This module provides a command-line interface for interacting with the chatbot.

### `main()`

When run as a script, this function starts the CLI, where the user can interact with the MedBot agent in the terminal.

*   **Usage**:
    ```bash
    python src/data_retrieve.py
    ```

---

## 6. Helpers (`src/helpers.py`)

This module provides utility functions used across the application.

### `load_config(config_path: str = 'config/config.yaml') -> dict`

Loads the main YAML configuration file.

*   **Parameters**:
    *   `config_path` (str): The path to the YAML configuration file.
*   **Returns**:
    *   `dict`: A dictionary containing all configuration settings.

### `create_vectorstore(index_name: str)`

Checks if a Pinecone index with the given name exists and creates it if it doesn't.

*   **Parameters**:
    *   `index_name` (str): The name of the Pinecone index to create.

