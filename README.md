# MedBot: A Medical Chatbot

MedBot is a conversational AI project that acts as a helpful medical assistant. Users can ask medical-related questions, and the chatbot will provide answers based on a knowledge base of medical books and documents. It features a user-friendly web interface and a command-line interface.

**Disclaimer:** This chatbot is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Features

- **Conversational AI:** Engage in natural conversations about medical topics.
- **Retrieval-Augmented Generation (RAG):** Provides answers based on a curated knowledge base of medical texts.
- **Source Citation:** Cites the book and page number for the information it provides.
- **Query Refinement:** Automatically rewrites user questions for more accurate search results.
- **Relevance Filtering:** Ensures that the information used to answer questions is relevant to the user's query.
- **Web & CLI Interfaces:** Interact with the chatbot through a Streamlit web app or a command-line interface.

## How it Works

MedBot is built using a Retrieval-Augmented Generation (RAG) architecture with LangChain and Pinecone.

1.  **Indexing:** A collection of medical PDFs (located in the `/data` directory) is processed by the `index.py` script. The text is extracted, split into smaller chunks, and then converted into numerical representations (embeddings) using the `text-embedding-3-small` model from OpenAI. These embeddings are stored in a Pinecone vector store.

2.  **User Interaction:** The user asks a question through either the Streamlit web app (`app.py`) or the command-line interface (`retrieve.py`).

3.  **Agent Execution:** The query is sent to a LangGraph agent which orchestrates the following steps:
    a. **Query Rewriting:** The agent first uses `gpt-4o-mini` to rewrite the user's query to be more specific and searchable.
    b. **Retrieval:** The rewritten query is used to search the Pinecone vector store and retrieve the most relevant text chunks from the medical documents.
    c. **Relevance Filtering:** The retrieved documents are then filtered for relevance to the original query using another call to `gpt-4o-mini`.
    d. **Generation:** The filtered, relevant text chunks are passed to the `gpt-4o-mini` model along with the original question and a detailed system prompt. The model uses this context to generate a concise and helpful answer, complete with citations.

4.  **Streaming Response:** The agent's intermediate steps and the final answer are streamed back to the user interface in real-time.

## Setup and Installation

Follow these steps to set up and run the chatbot on your local machine.

### Prerequisites

- Python 3.8 or higher
- An API key from [OpenAI](https://openai.com/api/)
- An API key from [Pinecone](https://www.pinecone.io/)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/MedBot-A-Medical-Chatbot.git
cd MedBot-A-Medical-Chatbot
```

### 2. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a file named `.env` in the root of the project directory and add your API keys:

```
OPENAI_API_KEY="your_openai_api_key"
PINECONE_API_KEY="your_pinecone_api_key"
```

### 4. Create the Knowledge Base

Before you can use the chatbot, you need to process the data and create the vector store. Run the following command:

```bash
python index.py
```

This script will read the PDFs from the `/data` folder, process them, and upload them to your Pinecone index.

## Usage

You can interact with MedBot through the web interface or the command-line interface.

### Web Interface

To run the Streamlit web app, use the following command:

```bash
streamlit run app.py
```

This will open the web app in your browser.

### Command-Line Interface

To start the chatbot in your terminal, run:

```bash
python retrieve.py
```

The chatbot will be ready to answer your questions. To exit the chatbot, type `exit`.

## Evaluation

The performance of the chatbot can be evaluated using the `test_rag.py` script. This script runs a series of predefined questions against the chatbot and compares the generated answers to a set of "ground truth" answers (located in `ground_truths/`) to measure the quality and accuracy of the responses.

## Data

The knowledge base for this chatbot is sourced from the following PDF documents located in the `/data` directory:

- `biology.pdf`
- `encyclopedia_of_medicine.pdf`
- `health_safety_and_nutrition.pdf`
- `human_nutrition.pdf`
- `nursing_fundamentals.pdf`
- `nursing_skills.pdf`

## Key Dependencies

This project relies on several key Python libraries:

- `langchain`: A framework for developing applications powered by language models.
- `streamlit`: A framework for building interactive web apps.
- `pypdf`: A library for reading and extracting text from PDF files.
- `python-dotenv`: For managing environment variables.
- `langchain-pinecone`: An integration for using Pinecone vector stores with LangChain.
- `langchain-openai`: An integration for using OpenAI models with LangChain.
- `langchain-community`: Community-contributed components for LangChain.

For a full list of dependencies, please see the `requirements.txt` file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.