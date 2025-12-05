from pinecone import Pinecone
from pinecone import ServerlessSpec
import yaml

# Used to reuse the loading config file.
def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

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