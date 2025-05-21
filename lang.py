import os
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from typing import List

from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

class ToyRetriever(BaseRetriever):
    documents: List[Document]
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        matching_documents = []
        for doc in self.documents:
            if len(matching_documents) >= self.k:
                break
            if query.lower() in doc.page_content.lower():
                matching_documents.append(doc)
        return matching_documents


if __name__ == "__main__":
    # 讀取你的 .txt 檔案並轉換為 Document 物件
    file_path = "C:/Users/Serena Li/OneDrive/Desktop/實驗室/team/【愛健康│理財生活通】陳亮恭醫師談「在家養老做得到嗎？」.txt" # 替換為你的檔案路徑
    loaded_docs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 將整個檔案內容視為一個 Document
            loaded_docs.append(Document(page_content=content))
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。請檢查檔案路徑是否正確。")
        exit() # 程式終止，因為找不到文件

    # 將 loaded_docs 賦值給 docs
    docs = loaded_docs

    # 問題
    question_template = "什麼是腦年齡？"
    question = question_template # 這裡可以直接使用帶有語言指令的問題

    # 建立檢索器
    retriever = ToyRetriever(documents=docs, k=2)

    # 檢索相關文件
    retrieved_docs = retriever._get_relevant_documents(query=question, run_manager=None)

    # 整合文件成字串
    docs_text = "".join(doc.page_content for doc in retrieved_docs)

    # 定義系統提示 (將繁體中文指令放在最前面並加強語氣)
    system_prompt = """你是一個問答助理。
    **請務必使用繁體中文回答，這非常重要。**
    請使用以下提供的上下文來回答問題。
    如果你不知道答案，就說你不知道。
    請最多使用三句話，並保持答案簡潔。
    Context: {context}:"""

    # 套入文件內容
    system_prompt_fmt = system_prompt.format(context=docs_text)

    # 建立 LLM 模型 (使用 Ollama)
    llm = OllamaLLM(model="gemma:7b") # 使用你選擇的模型

    # 建立 RetrievalQA 鏈
    # 注意：LangChain 的 RetrievalQA 預設會創建一個包含 {question} 和 {context} 的 prompt。
    # 我們的 system_prompt 會在底層被結合進去。
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # 發送提問並取得回答
    answer = qa.invoke({"query": question})["result"]

    print("\n問題：", question)
    print("\n回答：", answer)