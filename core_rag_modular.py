
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import os

# Function to load files
def load_documents_from_files(file_paths):
    documents = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        elif ext in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            continue

        docs = loader.load()
        documents.extend(docs)
        print(f"Loaded {len(docs)} document(s) from {file_path}")

    return documents

# Function to chunk documents
def chunk_documents(docs, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks

# Function to create Chroma vectorstore
def create_chroma_vectorstore(docs, persist_directory="chromaDB", model_name="BAAI/bge-base-en-v1.5"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="smartdoc_collection_bge"
    )
    return vectorstore

# Function to generate answer from retrieved chunks
def generate_answer(query, retrieved_docs, model_name="tinyllama"):
    llm = Ollama(model=model_name)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    You are a helpful assistant. Use the context below to answer the question accurately.

    Context:
    {context}

    Question: {query}
    Answer:"""

    return llm.invoke(prompt)
