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
    temperature=0.2,
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

# 設定TextLoader
#from langchain_community.document_loaders import TextLoader

template = ("<s>[INST]<<SYS>> \n" +
           "你是一個寫作專家，專精於改寫文章。 \n" +
           "請幫我分析本文後，將本文重新改寫，改寫方向是簡短地摘要隱含的關鍵訊息，\n" +
           "去掉不必要的贅字，只保留能表達本文意思的文句即可，例如「以達....之目的」， \n" +
           "「確保....」，「符合....」之類的文字都不需要， \n" +
           "改寫後的內容將會做為示範，所以要很慎重處理文字，允許適度加入你的想法。\n" +
           "<</SYS>>\n\n" +
           "本文: {text_input}\n" +
           "[/INST]</s>")

# 建立主視窗
window = tk.Tk()
window.title("TAIDE 應用程式 -- 文字改寫")

# 建立多行文字方塊
text_box1 = tk.Text(window, height=5, width=90)

text_box2 = tk.Text(window, height=50, width=100)

# 建立標籤
label1 = tk.Label(window, text="Before：")
label2 = tk.Label(window, text="After ：")

# 建立事件處理函式，當按鈕被點擊時觸發
def button_click():
    data = text_box1.get("1.0", "end-1c")  # 取得文字方塊內容
    text_box2.delete("1.0", "end")  # 清空 input2 的內容
    
    article = data.replace('\n\n','')

    #messagebox.showinfo('網頁', article)

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

    data = text_box1.get("1.0", "end-1c")  # 取得文字方塊內容
    article = data.replace('\n\n','')
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
