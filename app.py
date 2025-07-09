from core_rag_modular import *

docs = load_documents_from_files(["ml_test_doc.pdf"])
chunks = chunk_documents(docs)
vectorstore = create_chroma_vectorstore(chunks)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
results = retriever.get_relevant_documents("What is supervised learning?")
answer = generate_answer("What is supervised learning?", results)

print(answer)
