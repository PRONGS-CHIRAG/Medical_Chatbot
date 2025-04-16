from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from application.config import OPENAI_API_KEY,EMBEDDING_MODEL
import shutil
import json
import uuid
import numpy as np
import faiss
from Bio import Entrez
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def create_vector_store(docs, persist_dir="embeddings/faiss_index"):
    """For PDF"""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_dir)
    shutil.make_archive("faiss_index","zip","embeddings/faiss_index")

def load_vector_store(persist_dir="embeddings/faiss_index"):
    """For PDF"""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

#Search & Fetch PubMed abstracts
def search_pubmed(query, max_results=5):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    return record["IdList"]

def fetch_pubmed_details(pmid_list):
    ids = ",".join(pmid_list)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
    return handle.read()

def get_pubmed_articles(query, max_results=5):
    ids = search_pubmed(query, max_results)
    abstracts = fetch_pubmed_details(ids)
    return abstracts, ids

#Chunk using LangChain splitter
def clean_and_chunk(abstracts_list, pmid_list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for text, pmid in zip(abstracts_list, pmid_list):
        chunks = splitter.create_documents([text], metadatas=[{"source": "PubMed", "id": pmid}])
        all_chunks.extend(chunks)
    return all_chunks



#Create FAISS index
def build_faiss_index(chunks,save_path="embeddings/faiss_index",openai_api_key=OPENAI_API_KEY):
    #Initialising embedding model
    embedding = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=OPENAI_API_KEY)
    docs = [Document(page_content=chunk.page_content,metadata=chunk.metadata) for chunk in chunks]
    # Create FAISS index via LangChain
    vectorstore = FAISS.from_documents(docs, embedding)
    # Save vectorstore to disk
    vectorstore.save_local(save_path)
    shutil.make_archive("faiss_index","zip","embeddings/faiss_index")
    return vectorstore

def load_new_vectorstore(save_path="embeddings/faiss_index"):
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def dynamic_vs_creation():
    query = "Get documents related to covid-19, common cold, H1N1 and influenza"
    Entrez.email = "chiragnvijay@gmail.com"
    abstract_text, ids = get_pubmed_articles(query, max_results=30)
    abstracts = abstract_text.split("\n\n")  # or however PubMed separates them
    chunks = clean_and_chunk(abstracts, ids)
    print("Number of chunks :", len(chunks))
    print("Building FAISS index now")
    vectorstore = build_faiss_index(chunks,save_path="embeddings/faiss_index",openai_api_key=OPENAI_API_KEY)
    
    

