# Check if this content is in Pinecone
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


pc = Pinecone()
index = pc.Index("medbot")

# Search directly for the content
client = OpenAI()
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input="Safety Data Sheets (SDS), formerly referred to as Material Safety Data Sheets",
    dimensions=700
).data[0].embedding

results = index.query(
    vector=query_embedding, 
    top_k=10,
    include_metadata=True  # ← THIS IS CRITICAL
)

for match in results['matches']:
    print(f"Score: {match['score']}")
    print(f"Text: {match.get('metadata', {}).get('text', 'NO TEXT FOUND')[:200]}")
    print(f"Source: {match.get('metadata', {}).get('source', 'NO SOURCE')}")
    print(f"Page: {match.get('metadata', {}).get('page', 'NO PAGE')}")
    print("-" * 80)



# # Check if ANY result contains SDS content
# for match in results['matches']:
#     print(f"Score: {match['score']}")
#     print(match)