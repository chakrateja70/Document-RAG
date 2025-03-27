from sentence_transformers import SentenceTransformer
import chromadb

# Load the embedding model
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

# Initialize ChromaDB client and get the collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="rag_chunks")

# Retrieve all stored chunks and their embeddings
results = collection.get(include=["documents", "embeddings"])

# Print chunks and their corresponding embeddings
for i, (chunk, embedding) in enumerate(zip(results["documents"], results["embeddings"])):
    print(f"\n🔹 Chunk {i+1}: {chunk}")
    print(f"🔸 Embedding: {embedding}\n")
