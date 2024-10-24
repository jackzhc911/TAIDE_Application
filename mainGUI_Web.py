import tkinter as tk
from tkinter import messagebox
import asyncio

# 設定LLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

model_path = ".\\taide\\taide-7b-a.2-q4_k_m.gguf"

llm = LlamaCpp(
    max_tokens = 2048,
    model_path=model_path,
    temperature=0.1,
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

# 設定WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
template = ("<s>[INST]<<SYS>> \n" +
           "你是一個資訊安全方面的情資專家，請參考彰化縣地方稅務局的現況:\n" +
           "，並分析本文與資訊安全的內容後，簡短地摘要隱含的關鍵訊息，\n" +
           "包含但不限於受影響的產品名稱、產品類型、漏洞名稱、漏洞編號、影響系統、修補方法、相關網站等，\n" +
           "並在每個關鍵訊息前從1開始加上編號，可以適時適度加入你的意見及建議。\n" +
           "<</SYS>>\n\n" +
           "本文: {text_input}\n" +
           "[/INST]</s>")

# 建立主視窗
window = tk.Tk()
window.title("TAIDE 應用程式 -- 網站情資分析")

# 建立多行文字方塊
text_box1 = tk.Text(window, height=1, width=90)
text_box1.insert("1.0", "https://www.ithome.com.tw/news/162590")  # 設定預設值

text_box2 = tk.Text(window, height=50, width=100)

# 建立標籤
label1 = tk.Label(window, text="URL：")
label2 = tk.Label(window, text="RESULT：")

# 建立事件處理函式，當按鈕被點擊時觸發
def button_click():
    url = text_box1.get("1.0", "end-1c")  # 取得文字方塊內容
    text_box2.delete("1.0", "end")  # 清空 input2 的內容
    
    loader = WebBaseLoader(url)
    data = loader.load()
    article = data[0].page_content.replace('\n\n','')

    messagebox.showinfo('網頁', article)

    fact_extraction_prompt = PromptTemplate(
        input_variables=['text_input'],
        template = template
    )

    fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt)
    result = fact_extraction_chain.invoke(article)

    text_box2.insert(tk.END, result['text'])

# stream functions
import threading
async def stream_to_textbox(textbox):

    url = text_box1.get("1.0", "end-1c")  # 取得文字方塊內容
    loader = WebBaseLoader(url)
    data = loader.load()
    article = data[0].page_content.replace('\n\n','')
    fact_extraction_prompt = PromptTemplate(
        input_variables=['text_input'],
        template = template
    )
    fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt)
    #fact_extraction_chain = fact_extraction_prompt | llm | JsonOutputParser()

    #async for chunk in llm.astream("你好，你是誰"):
    async for chunk in fact_extraction_chain.astream(article):
        textbox.insert(tk.END, chunk)
        textbox.see(tk.END)
        await asyncio.sleep(0)  # Yield control to the event loop

def start_streaming():
    # Create a new event loop and set it as the current event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run the event loop in a separate thread to avoid freezing the Tkinter GUI
    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(stream_to_textbox(text_box2))
        loop.close()
    
    threading.Thread(target=run_loop).start()

# 建立按鈕
#button = tk.Button(window, text="開始", command=start_streaming)
button = tk.Button(window, text="開始", command=button_click)

# 放置元件到視窗中
label1.grid(row=0, column=0, sticky="w", padx=10, pady=10)
text_box1.grid(row=0, column=1, padx=10, pady=10)
button.grid(row=0, column=2, padx=10, pady=10)

label2.grid(row=1, column=0, sticky="w", padx=10, pady=10)
text_box2.grid(row=1, column=1, columnspan=2, padx=10, pady=10)


# 運行主程式
window.mainloop()
