import os
import chromadb
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_community.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddings,
# )
from sentence_transformers import SentenceTransformer

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


def split_chunks(pages: List[Document]) -> List[Document]:
    """
    Split documents into chunks of text.

    Args:
        pages (List[Document]): List of loaded document pages

    Returns:
        List[Document]: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(pages)
    return chunks


def embed_documents(chunks: List[Document]) -> None:
    # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # document_embeddings = embedding_function.embed_documents(
    #     [split.page_content for split in chunks]
    # )
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    chunk_texts = [chunk.page_content for chunk in chunks]
    document_embeddings = model.encode(chunk_texts)
    
    return document_embeddings


def chroma_db(chunks: List[Document], embeddings: List[List[float]]) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError(f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name="rag_chunks")
    except Exception:
        pass
    
    # Create new collection with 1024 dimensionality
    collection = client.create_collection(
        name="rag_chunks", 
        metadata={"dimensionality": 1024}
    )

    # Add chunks to the collection
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Generate a more unique chunk ID
        chunk_id = f"chunk_{i}_{hash(chunk.page_content)}"
        
        collection.add(
        ids=[chunk_id],
        embeddings=[embedding],  
        documents=[chunk.page_content],
        metadatas=[{"text": chunk.page_content}]  # Ensure metadata is stored
    )

        
    print(f"Stored {len(chunks)} embeddings in ChromaDB with dimension 1024.")