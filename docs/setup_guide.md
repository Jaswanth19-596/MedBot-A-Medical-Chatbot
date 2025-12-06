# MedBot Setup Guide

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Environment Setup](#3-environment-setup)
4. [Data Preparation and Indexing](#4-data-preparation-and-indexing)
5. [Running the Application](#5-running-the-application)
6. [Running Evaluations](#6-running-evaluations)
7. [Troubleshooting](#7-troubleshooting)
8. [FAQ](#8-faq)

---

## 1. Prerequisites

Before you begin, ensure you have the following:

*   **Python**: Version 3.13.3.
*   **API Keys**:
    *   **OpenAI API Key**: For accessing GPT-4o-mini and embedding models.
    *   **Pinecone API Key**: For accessing your vector database.
*   **Git**: For cloning the repository.

## 2. Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/MedBot-A-Medical-Chatbot.git
    cd MedBot-A-Medical-Chatbot
    ```

2.  **Create a virtual environment**:
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    *(On Windows, use `venv\Scripts\activate`)*

3.  **Install the required packages**:
    The project dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## 3. Environment Setup

The application requires API keys and other configuration values to be stored in environment variables.

1.  **Create a `.env` file** in the root directory of the project.

2.  **Add the following key-value pairs** to the `.env` file, replacing the placeholder values with your actual credentials:
    ```env
    OPENAI_API_KEY="your-openai-api-key"
    PINECONE_API_KEY="your-pinecone-api-key"
    ```

    The application uses `python-dotenv` to automatically load these variables at runtime.

## 4. Data Preparation and Indexing

The chatbot's knowledge is derived from PDF documents you provide.

1.  **Place your data**:
    *   Add your medical PDF documents into the `data/` directory. The project is pre-configured with several medical textbooks.

2.  **Run the indexing pipeline**:
    The `src/data_indexing.py` script handles the entire data processing pipeline. This script will:
    *   Ingest the PDFs from the `data/` directory.
    *   Chunk the documents into smaller pieces.
    *   Create vector embeddings for each chunk.
    *   Upload the embeddings and metadata to the Pinecone `medbot` index.

    Execute the script from the root directory:
    ```bash
    python src/data_indexing.py
    ```
    This process may take some time depending on the number and size of your documents.

## 5. Running the Application

You can interact with MedBot through the web interface or a command-line interface.

### Web Interface

The user interface is built with Streamlit.

1.  **Ensure your virtual environment is active**:
    ```bash
    source venv/bin/activate
    ```

2.  **Run the Streamlit app**:
    The main application file is `app.py`.
    ```bash
    streamlit run app.py
    ```

3.  **Access the web interface**:
    Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`). You can now start asking medical questions.

### Command-Line Interface

For a terminal-based experience, you can use the CLI.

1.  **Ensure your virtual environment is active**:
    ```bash
    source venv/bin/activate
    ```

2.  **Run the CLI script**:
    ```bash
    python src/data_retrieve.py
    ```
    The chatbot will be ready to answer your questions in the terminal. To exit, type `exit`.

## 6. Running Evaluations

We use the RAGAS framework to evaluate the performance of the RAG pipeline.

1.  **Prepare Ground Truth Data**:
    *   Ensure your ground truth Q&A pairs are in JSON format and located in the `ground_truths/` directory. The project includes 129 pairs across 3 files.

2.  **Run the evaluation script**:
    The `evaluate.py` script will run the RAGAS evaluation on the dataset and save the results.
    ```bash
    python evaluate.py
    ```

3.  **Check the results**:
    The evaluation scores will be printed to the console and saved in a file named `ragas_evaluation_results.csv` in the root directory.

## 7. Troubleshooting

| Issue | Solution |
|---|---|
| **`ModuleNotFoundError`** | Make sure you have activated the virtual environment (`source venv/bin/activate`) and installed all dependencies (`pip install -r requirements.txt`). |
| **API Authentication Error** | Double-check that your `.env` file is correctly formatted and that your `OPENAI_API_KEY` and `PINECONE_API_KEY` are valid and have not expired. |
| **Pinecone Index Not Found**| Ensure you have successfully run the indexing script (`python src/data_indexing.py`) before running the main application. Verify the index name in `config/config.yaml` matches the one in your Pinecone account. |
| **Slow response times** | This could be due to network latency or rate limits on the OpenAI API. If the problem persists, check the OpenAI status page. |
| **PDF processing errors** | Ensure your PDFs are not corrupted and are text-based. Scanned image-based PDFs cannot be processed by the default data loader. |

## 8. FAQ

**Q: Why is there a rate limit on the web interface?**
**A:** The web interface includes a rate limit (e.g., 5 requests per 30 minutes) to manage API costs associated with the language model and to ensure fair usage for all users. The CLI does not have this limit.

**Q: Can I use a different LLM?**
**A:** Yes. You can change the model name in `config/config.yaml`. You may also need to adjust `src/agent.py` if the new model requires a different API structure or prompt format.

**Q: How can I add more documents?**
**A:** Simply add the new PDF files to the `data/` directory and re-run the indexing script: `python src/data_indexing.py`. The script is configured to upsert data, so existing embeddings will be updated.

**Q: What does a "temperature" of 0 mean?**
**A:** A temperature of 0 makes the LLM's output deterministic. It will always produce the most likely next word, which is ideal for a factual Q&A system to minimize creativity and randomness. You can configure this in `config/config.yaml`.

**Q: Can I change the chunk size?**
**A:** Yes, the chunk size and overlap can be modified in the `config/config.yaml` file. After changing, you will need to re-run the indexing pipeline to update the documents in your vector store.
