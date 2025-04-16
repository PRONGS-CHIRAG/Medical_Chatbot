from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from application.vector_store import *
from application.loaders import * 
from application.config import OPENAI_API_KEY, LLM_MODEL
import streamlit as st

def medical_chatbot(user_query: str) -> str:
    print("Starting to create vector store")
    docs = load_and_split_docs(path="data/medical_docs")
    #create_vector_store(docs, persist_dir="embeddings/faiss_index")
    #Downloads the embeddings folder locally
    #with open("faiss_index.zip", "rb") as f:
       # st.download_button("Download Embeddings", f, "faiss_index.zip")
    #print("Loading vector store")
    #vectorstore = load_vector_store()
    vectorstore2 = load_new_vectorstore()
   # retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever2 = vectorstore2.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold":0.75,"k":5})
    llm = ChatOpenAI(
        temperature=0.2,
        model_name=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    # Create your custom prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
     "You are a medical assistant. Use the **provided context** strictly to answer the user’s medical question. "
     "If the context is not sufficient, say 'I'm sorry, I couldn't find enough information.'\n\nContext:\n{context}"),
    ("human", "{question}")
    ])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever2,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    

    result = qa_chain.run(user_query)
    return result
    
