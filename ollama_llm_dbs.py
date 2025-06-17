import os
import pandas as pd
from bert_score import score
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA

# ---------- 讀取問答 ----------
def load_qa_from_txt(file_path: str) -> pd.DataFrame:
    questions = []
    answers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '?' in line:
                q, a = line.strip().split('?', 1)
                questions.append(q.strip())
                answers.append(a.strip())
    return pd.DataFrame({'Question': questions, 'Answer': answers})


# ---------- 建立向量資料庫 ----------
def build_vector_store(source_path: str, persist_dir: str = "chroma_db") -> Chroma:
    loader = TextLoader(source_path, encoding='utf-8')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
    return vector_store


# ---------- 問題+檢索 → LLM 回答（使用 invoke()） ----------
def rag_qa(vector_store: Chroma, query: str, llm_model: str = 'gemma:7b') -> str:
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOllama(model=llm_model),
        retriever=retriever,
        chain_type="stuff"
    )
    result = qa_chain.invoke({"query": query}) 
    return result.strip() if isinstance(result, str) else str(result)


# ---------- RAG + BERTScore ----------
def use_rag_with_bertscore(data: pd.DataFrame, vector_store: Chroma, llm_model: str = 'gemma:7b') -> pd.DataFrame:
    records = []
    system_instruction = "請完全使用繁體中文，並且不要使用條列式，使用不超過兩句話回答以下問題。"
    for idx, row in data.iterrows():
        try:
            gen_answer = rag_qa(vector_store, row['Question'], llm_model)
        except Exception as e:
            print(f"❌ 回答失敗 Q{idx+1}: {e}")
            gen_answer = "(Error)"

        try:
            _, _, f1 = score([gen_answer], [row['Answer']], lang='zh', rescale_with_baseline=True)
            bert_f1 = round(f1[0].item(), 3)
        except Exception as e:
            print(f"⚠️ BERTScore 計算錯誤：{e}")
            bert_f1 = 0.0

        records.append({
            'Question': row['Question'],
            'GroundTruth': row['Answer'],
            'Generated': gen_answer,
            'BERT_F1': bert_f1
        })

        print(f"Q{idx+1}: {row['Question']}")
        print(f"  ▶ 正確答案: {row['Answer']}")
        print(f"  ▶ RAG 回答: {gen_answer}")
        print(f"  ▶ 語意相似度 BERT F1: {bert_f1:.3f}\n")

    return pd.DataFrame(records)


# ---------- 主程式 ----------
if __name__ == '__main__':
    test_path = "C:/Users/Serena Li/OneDrive/Desktop/實驗室/team/test.txt"     # 問題與正解
    source_path = "C:/Users/Serena Li/OneDrive/Desktop/實驗室/team/【愛健康│理財生活通】陳亮恭醫師談「你知道自己的腦年齡嗎？」.txt" # 教材來源
    persist_path = "chroma_db"

    # 載入資料與建庫
    df = load_qa_from_txt(test_path)
    vector_store = build_vector_store(source_path, persist_path)

    # 啟動 RAG + 評估
    result_df = use_rag_with_bertscore(df, vector_store, llm_model='gemma:7b')
