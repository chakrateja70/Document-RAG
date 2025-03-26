# Intelligent Document Retrieval System

## Project Overview

A Retrieval-Augmented Generation (RAG) system that enables semantic document analysis and intelligent question-answering across multiple document formats.

## Project Aim

- Ingest and process documents from PDF, DOCX, and TXT formats
- Perform semantic embedding using advanced transformer models
- Create a vector database for efficient document retrieval
- Generate context-aware answers using local language models

## Technologies & Dependencies

### Core Libraries
- LangChain
- ChromaDB
- SentenceTransformers
- Ollama
- python-dotenv

### Supported Document Types
- PDF
- DOCX
- TXT

## Key Components

### Document Processing
- Multi-format document loader
- Recursive text chunking
- Semantic embedding generation

### Retrieval Mechanism
- Vector-based document similarity search
- Top-k chunk retrieval
- Context-aware query processing

## Prerequisites

- Python 3.8+
- Ollama
- Hugging Face SentenceTransformers

## Installation

```bash
# Clone the repository
git clone https://github.com/chakrateja70/Document-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Model Recommendations

- If you want better accuracy and richer responses → Llama 3:8B
- If you need faster responses and lower memory usage → Mistral 7B

```bash
# Pull recommended models
ollama pull llama3:8b
ollama pull mistral:7b
```

## Configuration

### Environment Variables
- Create a `.env` file for any sensitive configurations
- Supports customization of embedding models and retrieval parameters

## Usage Example

```python
# Load documents
documents = load_documents('./documents')
chunks = split_chunks(documents)
embeddings = embed_documents(chunks)

# Create vector database
collection = chroma_db(chunks, embeddings)

# Set up RAG chain
rag_chain = setup_rag_chain(collection)

# Query the system
response = rag_chain.invoke("Your query here")
print(response)
```

## Customization Points

- Embedding Model: Currently using `jinaai/jina-embeddings-v3`
- Language Model: Configurable (Llama 3:8B or Mistral 7B recommended)
- Chunk Size: Configurable in `split_chunks()` method
- Retrieval Strategy: Adjustable top-k results

## Potential Improvements
- Support for more document formats
- Enhanced error handling
- Performance optimization
- Additional embedding models
- Web/CLI interface

## License
MIT License

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push and submit pull request