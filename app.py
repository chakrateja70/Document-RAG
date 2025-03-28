from doc_embed import (
    load_documents,
    split_chunks,
    embed_documents,
    chroma_db,
    setup_rag_chain,
)


def main():
    folder_path = "files/"
    pages = load_documents(folder_path)
    print(f"Loaded {len(pages)} documents")

    chunks = split_chunks(pages)
    print(f"Split the document into {len(chunks)} chunks.")

    embeddings = embed_documents(chunks)
    print(f"Embedded {len(embeddings)} chunks.")

    # Store in ChromaDB
    collection = chroma_db(chunks, embeddings)

    # Setup RAG chain
    rag_chain = setup_rag_chain(collection)

    # Example query
    query="Your query here?"
    response = rag_chain.invoke(query)
    print(f"Query: {query}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
