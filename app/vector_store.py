from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from app.config import OPENAI_API_KEY, EMBEDDING_MODEL

def create_vector_store(docs, persist_dir="embeddings/faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_dir)

def load_vector_store(persist_dir="embeddings/faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(persist_dir, embeddings)
    return vectorstore
