from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def create_pdf_vector_store(file_name,model_name,chunk_size=500,chunk_overlap=20):
    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model=model_name)
    loader = PyPDFLoader(file_name)
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                 chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
    return vectorstore


if __name__ == "__main__":
    embedding_model_name='nomic-embed-text'
    model_name='llama3'
    file_name='data/tuning_sgmcmc.pdf'
    ollama = Ollama(base_url='http://localhost:11434',model=model_name)
    vector_store=create_pdf_vector_store(file_name,embedding_model_name)
    qachain=RetrievalQA.from_chain_type(ollama, retriever=vector_store.as_retriever())
    question='describe the methodology of the paper'
    res = qachain.invoke({"query": question})
    print(res['result'])  