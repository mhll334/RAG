import os
import platform
import ollama
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rouge_score import rouge_scorer
from bert_score import score

# ── 基本設定 ──
DOCUMENT_PATH = "C:/Users/Serena Li/OneDrive/Desktop/實驗室/team/【愛健康│理財生活通】陳亮恭醫師談「你知道自己的腦年齡嗎？」.txt"
QUESTION_PATH = "C:/Users/Serena Li/OneDrive/Desktop/實驗室/team/test.txt"  # 格式：每行為「問題？正確答案」
MODELS = ["mistral", "llama3:8b", "gemma:7b", "taide-medicine-qa-tw-q6"]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_RETRIEVER_K = 5
PROMPT = """請根據以下內容以繁體中文作答，不得加入未提及的資訊。\n\n{context}\n\n問題：{question}\n回答："""

# ── 資料載入與向量化 ──
def load_and_split_documents(path: str):
    docs = TextLoader(path, encoding="utf-8").load()
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(docs)

def build_vectorstore(docs, collection_name="rag-collection"):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return Chroma.from_documents(docs, collection_name=collection_name, embedding=embeddings)

def get_retriever(vs):
    k = min(MAX_RETRIEVER_K, len(vs._collection.get()["metadatas"]))
    return vs.as_retriever(search_kwargs={"k": k})

def create_rag_chain(retriever, model_name: str):
    llm = ChatOllama(model=model_name, temperature=0.0)
    prompt = PromptTemplate.from_template(PROMPT)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt})

# ── 問答載入 ──
def load_questions(filepath):
    qas = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if "?" in line:
                q, a = line.strip().split("?")
                qas.append((q + "?", a.strip()))
    return qas

# ── 評估指標 ──
def compute_metrics(pred, ref):
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r = rouge.score(ref, pred)
    P, R, F1 = score([pred], [ref], lang="zh", verbose=False)
    return {
        "rouge1": r['rouge1'].fmeasure,
        "rouge2": r['rouge2'].fmeasure,
        "rougeL": r['rougeL'].fmeasure,
        "bert_f1": F1[0].item()
    }

# ── 主流程 ──
def evaluate_all():
    docs = load_and_split_documents(DOCUMENT_PATH)
    vs = build_vectorstore(docs)
    retr = get_retriever(vs)
    qas = load_questions(QUESTION_PATH)

    results = []
    for model in MODELS:
        print(f"🔍 Evaluating model: {model}")
        chain = create_rag_chain(retr, model)
        for q, gold in qas:
            res = chain.invoke({"query": q})
            pred = res["result"]
            metrics = compute_metrics(pred, gold)
            results.append({
                "model": model,
                "question": q,
                "reference": gold,
                "prediction": pred,
                **metrics
            })

    df = pd.DataFrame(results)
    df.to_csv("rouge_bert_evaluation.tsv", sep="\t", index=False)
    print("✅ 評估結果已儲存至 rouge_bert_evaluation.tsv")

if __name__ == "__main__":
    evaluate_all()
