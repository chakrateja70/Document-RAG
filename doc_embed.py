import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def load_documents(folder_path: str) -> List[Document]:
    """
    Load documents from a specified folder.
    Supports PDF, DOCX, and TXT file types.
    
    Args:
        folder_path (str): Path to the folder containing documents
    
    Returns:
        List[Document]: List of loaded document pages
    """
    pages = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        pages.extend(loader.load())
    return pages

def main():
    # Verify API key
    Pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not Pinecone_api_key:
        raise ValueError("Pinecone API key not found in environment variables")

    # Load documents
    folder_path = "files/"
    pages = load_documents(folder_path)
    print(f"Loaded {len(pages)} documents from the folder.")

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split the document into {len(chunks)} chunks.")
    
    index_name = "doc-rag"

    # Prepare embedding function
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store documents in Pinecone using PineconeVectorStore
    vectordb = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_function,
        index_name=index_name,
    )

    print("Data successfully stored in Pinecone!", vectordb)

    print(f"Data successfully stored in Pinecone under index '{index_name}'!")

    return vectordb

if __name__ == "__main__":
    main()