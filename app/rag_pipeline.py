from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from app.vector_store import load_vector_store
from app.config import OPENAI_API_KEY, LLM_MODEL

def medical_chatbot(user_query: str) -> str:
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        temperature=0.2,
        model_name=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain.run(user_query)
    return result
