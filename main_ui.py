import os
import platform
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── 1. config ──
DOCUMENT_DIR = "C:/Users/Serena Li/OneDrive/Desktop/vscode/with_punctuation" 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_RETRIEVER_K = 5
DEFAULT_MODEL = "mistral"

# ── 2. utils ──
def setup_chinese_font():
    import matplotlib
    if platform.system() == 'Windows':
        matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
    elif platform.system() == 'Darwin':
        matplotlib.rcParams['font.family'] = 'Heiti TC'
    else:
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK TC'
    matplotlib.rcParams['axes.unicode_minus'] = False

# ── 3. data_loader ──
@st.cache_data
def load_and_split_documents_from_dir(directory: str):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            all_docs.extend(splitter.split_documents(docs))
    return all_docs

# ── 4. index_builder ──
@st.cache_resource
def build_vectorstore(_docs, collection_name="rag-collection"):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return Chroma.from_documents(
        documents=_docs,
        collection_name=collection_name,
        embedding=embeddings
    )

def get_retriever(vs):
    total = len(vs._collection.get()["metadatas"])
    k = min(MAX_RETRIEVER_K, total)
    return vs.as_retriever(search_kwargs={"k": k})

# ── 5. rag_pipeline ──
PROMPT = """請根據以下內容完全以繁體中文作答，不可自行加入未提及的資訊。

{context}

問題：{question}
回答："""
@st.cache_resource
def create_rag_chain(_retriever, model_name: str):
    llm = ChatOllama(model=model_name)
    prompt = PromptTemplate.from_template(PROMPT)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

# ── Streamlit UI ──
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("📚 健康知識問答系統")

# 1. 準備資源
docs = load_and_split_documents_from_dir(DOCUMENT_DIR)
vs   = build_vectorstore(docs)
retr = get_retriever(vs)

# 2. 側欄設定
model = st.sidebar.selectbox("選擇模型", ["mistral", "llama3:8b", "gemma:7b", "taide-medicine-qa-tw-q6"], index=0)

# 3. 使用者輸入
question = st.text_area("請輸入您的問題：", height=150)
if st.button("產生回答"):
    with st.spinner("模型思考中..."):
        chain = create_rag_chain(retr, model_name=model)
        res   = chain.invoke({"query": question})
    st.subheader("📝 回答")
    st.write(res["result"])
    if res["source_documents"]:
        st.subheader("📑 依據來源")
        used_files = set()
        for doc in res["source_documents"]:
            path = doc.metadata.get('source', '')
            filename = os.path.splitext(os.path.basename(path))[0]
            used_files.add(filename)
        for name in sorted(used_files):
            st.write(f"- {name}")

