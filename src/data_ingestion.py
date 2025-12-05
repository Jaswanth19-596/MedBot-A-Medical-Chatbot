from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document



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