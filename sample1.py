from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.chains.question_answering import load_qa_chain

# 載入PDF檔
doc_reader = PdfReader('.\\taide\\底價訂定及價格偏低處理1120116.pdf')
raw_text = ''
for i, page in enumerate(doc_reader.pages):
  text = page.extract_text()
  if text:
    raw_text += text

# 測試是否載入成功
print(raw_text)

# 設定文句分割，200個字為單位，可覆蓋10個字
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(separator='\n', chunk_size=200, chunk_overlap = 10)

# 開始分割並顯示結果
texts = text_splitter.split_text(raw_text)
print(len(texts))
print(texts[0])

# 設定嵌入模型，用於計算文句的向量值
embeddings = SentenceTransformerEmbeddings(model_name="distiluse-base-multilingual-cased-v1")

# 設定向量資料庫
docsearch = FAISS.from_texts(texts, embeddings)

# 測試向量資料庫搜尋功能並顯示結果
#docs = docsearch.similarity_search(query)
#print(docs[0])

# 載入對話用大語言模型 -- TAIDE
model_path = ".\\taide\\taide-7b-a.2-q4_k_m.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

# 設定對話鏈，文件整理模式 stuff，單文件夠用了
chain = load_qa_chain(llm, chain_type="stuff")
#chain.run(input_documents=docs, question=query)

# 開始前先打個招呼
query="你好"
llm(query)

# 進入對話迴圈
while True:
    user_input = input("請輸入問題（輸入 'exit' 以離開）：")
    if user_input.lower() == "exit":
        print("已退出迴圈。")
        break
    else:
        docs = docsearch.similarity_search(user_input)
        chain.run(input_documents=docs, question=user_input)


