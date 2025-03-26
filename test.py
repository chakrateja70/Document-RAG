# import chromadb

# def print_chroma_embeddings():
#     # Initialize ChromaDB client
#     client = chromadb.PersistentClient(path="./chroma_db")
    
#     # Get the collection
#     collection = client.get_collection(name="rag_chunks")
    
#     # Retrieve all data
#     results = collection.get(include=["documents", "embeddings"])
    
#     # Print details
#     for i, (document, embedding) in enumerate(zip(results['documents'], results['embeddings']), 1):
#         print(f"Chunk {i} Embedding:")
#         print(embedding)  # Print full embedding
#         print("-" * 50)
    
#     print(f"Total chunks: {len(results['documents'])}")
    
# if __name__ == "__main__":
#     print_chroma_embeddings()


import chromadb

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Get the collection
collection = client.get_collection(name="rag_chunks")

# Retrieve all embeddings
all_embeddings = collection.get(include=["embeddings"])



# Print or process embeddings
print(all_embeddings["embeddings"])
