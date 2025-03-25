from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from typing import List
from langchain_core.documents import Document
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


"""This function will load all documents from a folder.
It will use the appropriate loader for each file type.
Supported file types are .pdf, .docx, and .txt.
The function will return a list of Document objects.
Each Document object will contain the text of a single page.
The function will print a warning for unsupported file types."""

def load_documents(folder_path: str) -> List[Document]:
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


folder_path = "files/"
pages = load_documents(folder_path)
# print(pages[1])

#print(f"Loaded {len(pages)} documents from the folder.")

# This text splitter will split the text into chunks 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(pages)
#print(f"split the document into  {len(chunks)} chunks.")
# second_doc_chunks = [chunk for chunk in splits if chunk.metadata["source"] == pages[1].metadata["source"]]

# if second_doc_chunks:
#     print(second_doc_chunks[3000])

"""This code will embed the text chunks using the MiniLM model.
The embeddings will be stored in a list.
The list will contain one embedding for each chunk.
The embeddings can be used to compare the similarity of the chunks.
The embeddings can also be used to cluster the chunks.
The embeddings can be used for other natural language processing tasks."""

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
for chunk in chunks:
    embeddings = model.encode(chunk.page_content)
# print(embeddings)

