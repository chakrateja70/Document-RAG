import os
import chromadb
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

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
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(pages)
    return chunks


def embed_documents(chunks: List[Document]) -> None:
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    chunk_texts = [chunk.page_content for chunk in chunks]
    document_embeddings = model.encode(chunk_texts)

    return document_embeddings


def chroma_db(chunks: List[Document], embeddings: List[List[float]]) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})"
        )

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection if it exists
    try:
        client.delete_collection(name="rag_chunks")
    except Exception:
        pass

    # Create new collection with 1024 dimensionality
    collection = client.create_collection(
        name="rag_chunks", metadata={"dimensionality": 1024}
    )
    # Add chunks to the collection
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Generate a more unique chunk ID
        chunk_id = f"chunk_{i}_{hash(chunk.page_content)}"

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk.page_content],
            metadatas=[{"text": chunk.page_content}],  # Ensure metadata is stored
        )
    return collection


def retrieve_relevant_chunks(
    query: str, collection: chromadb.Collection, top_k: int = 10
) -> List[str]:
    """
    Retrieve relevant chunks based on query similarity.

    Args:
        query (str): User's query
        collection (chromadb.Collection): ChromaDB collection
        top_k (int, optional): Number of top similar chunks to retrieve. Defaults to 3.

    Returns:
        List[str]: List of relevant chunk texts
    """
    # Use SentenceTransformer to embed the query
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    query_embedding = model.encode([query])

    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    return results["documents"][0]


def setup_rag_chain(collection: chromadb.Collection):
    """
    Set up the Retrieval-Augmented Generation (RAG) chain.

    Args:
        collection (chromadb.Collection): ChromaDB collection

    Returns:
        Callable: RAG chain for querying
    """
    llm = Ollama(model="llama3:8b")

    # Define the RAG prompt template
    prompt_template = """Use the following context to answer the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Helpful Answer:"""

    prompt = PromptTemplate.from_template(prompt_template)

    # Create RAG chain
    def format_docs(docs):
        return "\n\n".join(docs)

    def retrieve_context(query):
        return retrieve_relevant_chunks(query, collection)

    rag_chain = (
        {"context": retrieve_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
