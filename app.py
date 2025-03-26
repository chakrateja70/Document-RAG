from doc_embed import load_documents, split_chunks, embed_documents, chroma_db

def main():
    folder_path = "files/"
    pages = load_documents(folder_path)
    print(f"Loaded {len(pages)} documents")
    
    chunks = split_chunks(pages)
    print(f"Split the document into {len(chunks)} chunks.")

    embeddings = embed_documents(chunks)
    
    # print("Finished embedding the chunks.",embeddings)
    
    chroma_db(chunks, embeddings)
    print("Finished creating the ChromaDB.")
    

if __name__ == "__main__":
    main()