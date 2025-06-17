import os
import pandas as pd
from bert_score import score
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
def build_vector_store(txt_path: str, persist_dir: str = "chroma_db") -> Chroma:
    loader = TextLoader(txt_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
    return vector_store


# ---------- 自定義 prompt + RAG 查詢 ----------
def rag_qa(vector_store: Chroma, query: str, llm_model: str = 'llama3:8b') -> str:
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})

    # 自定義 Prompt（繁體中文 + 非條列式 + 不超過兩句）
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "請根據以下資料，完全使用繁體中文並以完整句回答問題，不得使用條列式，回答請控制在兩句以內。\n\n"
            "【資料內容】:\n{context}\n\n"
            "【問題】:\n{question}\n\n"
            "【回答】："
        )
    )

    llm = ChatOllama(model=llm_model)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    result = qa_chain.invoke({"query": query})
    return result.get("result", "").strip()


# ---------- 評估 + 相似度 ----------
def use_rag_with_bertscore(data: pd.DataFrame, vector_store: Chroma, llm_model: str = 'gemma:7b') -> pd.DataFrame:
    records = []
    for idx, row in data.iterrows():
        try:
            gen_answer = rag_qa(vector_store, row['Question'], llm_model)
        except Exception as e:
            print(f"❌ 回答失敗 Q{idx+1}: {e}")
            gen_answer = "(Error)"

        try:
            # ⚠️ 不使用 rescale baseline，以防語意相似反而負分
            _, _, f1 = score([gen_answer], [row['Answer']], lang='zh')
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
    test_path = "C:/Users/Serena Li/OneDrive/Desktop/實驗室/team/test.txt"     # 問答資料
    source_path = "C:/Users/Serena Li/OneDrive/Desktop/實驗室/team/【愛健康│理財生活通】陳亮恭醫師談「你知道自己的腦年齡嗎？」.txt" # 建庫資料
    persist_path = "chroma_db"

    df = load_qa_from_txt(test_path)
    vector_store = build_vector_store(source_path, persist_path)
    result_df = use_rag_with_bertscore(df, vector_store, llm_model='llama3:8b')
