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

    # Store in ChromaDB
    collection = chroma_db(chunks, embeddings)

    # Setup RAG chain
    rag_chain = setup_rag_chain(collection)

    # Example query
    query = "Provide a detailed explanation of the 'Multi-Agent System for Web Search and Finance' project by Chakra Teja Polamarasetty, including its implementation and key components."
    response = rag_chain.invoke(query)
    print(f"Query: {query}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
