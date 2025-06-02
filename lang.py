from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

loader = TextLoader("C:/Users/Serena Li/OneDrive/Desktop/實驗室/team/【愛健康│理財生活通】陳亮恭醫師談「在家養老做得到嗎？」.txt", encoding = "utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap = 0)
doc_splits = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents = doc_splits,
    collection_name = "local-rag",
    embedding = NomicEmbeddings(model = "nomic-embed-text-v1.5", inference_mode = "local"),
)
retriever = vectorstore.as_retriever()

llm = ChatOllama(model = "mistral", temperature = 0)

rag_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = retriever,
    return_source_documents = True,
)

query = "請說明在家養老需要做什麼準備?"
result = rag_chain.invoke({"query": query}) 

print("Answer: ", result["result"])