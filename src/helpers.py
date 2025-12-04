from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone
from pinecone import ServerlessSpec

def load_and_filter_documents(data_path):

  # Creating the Directory Loader
  loader = DirectoryLoader(
      path = data_path,
      glob = "**/*.pdf",
      loader_cls=PyPDFLoader
    )

  # Loading the documents
  docs = loader.load()

  # FIltering the documents => Removing unnecessary metadata
  filtered_documents = []

  for doc in docs:
    metadata = {
        'book_name' : doc.metadata['source'].strip('data/'),
        'page' : doc.metadata['page']

    }
    filtered_document = Document(page_content = doc.page_content, metadata = metadata)
    filtered_documents.append(filtered_document)

  return filtered_documents


# Split the documents into smaller chunks
def split_text_into_chunks(documents):

  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 40)

  chunks = text_splitter.split_documents(documents)

  return chunks


def create_vectorstore(index_name):
  pc = Pinecone()
  if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 700,
        metric = 'cosine',
        spec = ServerlessSpec(
            cloud = "aws",
            region = "us-east-1"
        )
    )


def extract_context(docs):
  context = ""
  for doc in docs:
    context += doc.page_content + " \n"

  return context