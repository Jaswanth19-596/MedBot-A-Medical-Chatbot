from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.helpers import load_config

config = load_config()

chunk_size = config['chunk']['chunk_size']
chunk_overlap = config['chunk']['chunk_overlap']


# Split the documents into smaller chunks
def split_text_into_chunks(documents):

  text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)

  chunks = text_splitter.split_documents(documents)

  return chunks