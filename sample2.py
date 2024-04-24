from PyPDF2 import PdfReader
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.chains.question_answering import load_qa_chain


# 載入PDF檔
doc_reader = PdfReader('.\\taide\\需求規範書.pdf')
raw_text = ''
for i, page in enumerate(doc_reader.pages):
  text = page.extract_text()
  if text:
    raw_text += text

# 測試是否載入成功
# print(raw_text)

# 設定文句分割，200個字為單位，可覆蓋10個字
#text_splitter = CharacterTextSplitter.from_tiktoken_encoder(separator='\n', chunk_size=200, chunk_overlap = 10)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap = 5)

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
    max_tokens = 2000,
    model_path=model_path,
    temperature=0.1,
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

# 設定提示詞模版 -- LLMChain專用
#from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n You are an assistant tasked with improving Google search \
results. \n <</SYS>> \n\n [INST] Use Traditional Chinese to generate THREE Google search queries that \
are similar to this question. The output should be a numbered list of questions \
and each should have a question mark at the end: \n\n {question} [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with improving Google search \
results. Use Traditional Chinese to generate THREE Google search queries that are similar to \
this question. The output should be a numbered list of questions and each \
should have a question mark at the end: {question}""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)

# 設定提示詞模版 -- RetrievalQA專用
template = """
<s>[INST] <<SYS>>
你是一個幫助性強且用於精確檢查回答品質的Assistant，非常瞭解政府採購法，請利用文件內容，以繁體中文回答問題，如果你不知道答案就略過，請勿捏造答案。
<</SYS>>

文件內容是: {context}

Question: {question}
Helpful Answer:[/INST]
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain


# 設定對話鏈，文件整理模式 stuff，單文件夠用了
#llm_chain = LLMChain(prompt=prompt, llm=llm)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})

# no prompt
"""
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
    verbose=True
)
"""

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True,
    verbose=True
)

# 開始前先打個招呼
query="你好，對我打個招呼吧"
llm(query)

# 進入對話迴圈
while True:
    user_input = input("請輸入問題（輸入 'exit' 以離開）：")
    if user_input.lower() == "exit":
        print("已退出迴圈。")
        break
    else:
        #docs = docsearch.similarity_search(user_input)
        qa.invoke(user_input)


