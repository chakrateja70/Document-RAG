from app import main 

rag_chain = main()

query = "what are inputs required for Email connector ?"
response = rag_chain.invoke({"query": query})

answer = response.get("result", "No relevant information found in the knowledge base.")

print("Answer:", answer)
