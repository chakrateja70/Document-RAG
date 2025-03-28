Here’s the updated **README.md** with the command to run the application:  

---

# **Intelligent Document Retrieval System**

## **Project Overview**  
A Retrieval-Augmented Generation (RAG) system that enables semantic document analysis and intelligent question-answering across multiple document formats.

## **Project Aim**  
- Ingest and process documents from PDF, DOCX, and TXT formats  
- Perform semantic embedding using advanced transformer models  
- Create a vector database for efficient document retrieval  
- Generate context-aware answers using local language models  

## **Technologies & Dependencies**  

### **Core Libraries**  
- LangChain  
- ChromaDB  
- SentenceTransformers  
- Ollama (Optional, but recommended for local model execution)  
- python-dotenv  

### **Supported Document Types**  
- PDF  
- DOCX  
- TXT  

## **Key Components**  

### **Document Processing**  
- Multi-format document loader  
- Recursive text chunking  
- Semantic embedding generation  

### **Retrieval Mechanism**  
- Vector-based document similarity search  
- Top-k chunk retrieval  
- Context-aware query processing  

## **Prerequisites**  
- Python 3.8+  
- Hugging Face SentenceTransformers  
- Ollama (Optional but recommended for running local LLMs efficiently)  

## **Installation & Setup**  

```bash
# Clone the repository
git clone https://github.com/chakrateja70/Document-RAG

# Navigate to the project directory
cd Document-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## **Model Recommendations**  

- If you want better accuracy and richer responses → Llama 3:8B  
- If you need faster responses and lower memory usage → Mistral 7B  

```bash
# (Optional) Pull recommended models for local inference
ollama pull llama3:8b
ollama pull mistral:7b
```

**Note:** Installing Ollama is not mandatory. However, for better control over context and local execution of models, it's recommended to use Ollama with a GPT-based version.  

## **Configuration**  

### **Environment Variables**  
- Create a `.env` file for any sensitive configurations  
- Supports customization of embedding models and retrieval parameters  

## **Usage Example**  

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

## **Customization Points**  

- **Embedding Model:** Currently using `jinaai/jina-embeddings-v3`  
- **Language Model:** Configurable (Llama 3:8B or Mistral 7B recommended)  
- **Chunk Size:** Configurable in `split_chunks()` method  
- **Retrieval Strategy:** Adjustable top-k results  

## **Potential Improvements**  
- Support for more document formats  
- Enhanced error handling  
- Performance optimization  
- Additional embedding models  
- Web/CLI interface  

## **License**  
MIT License  

## **Contributing**  
1. Fork the repository  
2. Create a feature branch  
3. Commit changes  
4. Push and submit a pull request  

---

Now, the **installation and setup** section includes `python app.py` to run the application. 🚀