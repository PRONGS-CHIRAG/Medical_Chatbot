from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_docs(path="data/medical_docs"):
    loader = DirectoryLoader(path, glob="**/*.txt", show_progress=True)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    return docs
