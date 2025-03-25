from doc_embed import main
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# Load the vectordb object from doc_embed.py
vectordb = main()
def main():

    model_name = "google/flan-t5-large"
    llm_pipeline = pipeline("text2text-generation", model=model_name, tokenizer=model_name)

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    retriever = vectordb.as_retriever()

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    print("RAG pipeline successfully set up with Hugging Face model!")

    return rag_chain

if __name__ == "__main__":
    main()