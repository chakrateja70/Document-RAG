from sentence_transformers import SentenceTransformer
import chromadb

# Load the embedding model
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

# Initialize ChromaDB client and get the collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="rag_chunks")

# Define the query
query = "CERTIFICATIONS & TRAINING"
query_embedding = model.encode([query])

# Perform the search
results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=10
    )

# # Print the relevant chunks
# for i, doc in enumerate(results["documents"][0]):
#     print(f"Relevant Chunk {i+1}: {doc}")

# Print relevant chunks with cosine similarity > 0.8
for i, (doc, score) in enumerate(zip(results["documents"][0], results["distances"][0])):
    similarity = 1 - score  # Since ChromaDB returns distances, convert to similarity
    if similarity:
        print(f"🔹 Relevant Chunk {i+1}: {doc}")
        print(f"🔸 Cosine Similarity: {similarity:.4f}\n")
    else:
        print("No relevant chunks")