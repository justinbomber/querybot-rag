import os
import time
app_start_time = time.time()
import json
import logging
from datetime import datetime
from openai import AzureOpenAI
from embedding import document_activate, document_embedding, queryprocess, active_file, hello_query
import numpy as np

# 創建logs目錄（如果不存在）
if not os.path.exists('logs'):
    os.makedirs('logs')

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/main.log', encoding='utf-8'),
        logging.StreamHandler()  # 同時輸出到控制台
    ]
)

# 讀取openai_token.json
with open("openai_token.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# Azure OpenAI 設置
azure_endpoint = config["llm_model_config"]["azure_endpoint"]
api_key = config["llm_model_config"]["api_key"]
deployment_name = config["llm_model_config"]["deployment_name"]
api_version = config["llm_model_config"]["api_version"]
prompt_dir = "./"

logging.info(f"Azure OpenAI 配置：endpoint={azure_endpoint}, model={deployment_name}, version={api_version}")

# 初始化 Azure OpenAI 客戶端
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

def prompt_getter():
    if os.path.exists(prompt_dir + "/prompt.txt"):
        with open(prompt_dir + "/prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
<instruction>
只能用繁體中文回答
你是一個根據「上下文資訊的內容」回答使用者問題的問答系統。
上下文資訊為list，value為dict，dict會有兩個key，分別是similarity和content。
similarity是相似度，content是內容。
similarity越高，content越有可能是用戶的答案。
similarity越高，content越有可能是用戶的答案。
similarity越高，content越有可能是用戶的答案。
若有url則回答中一定要包含url
請務必僅依據提供的資料內容作答，不可憑空推測、擴充或發揮想像。
回答時請依以下規則嚴格執行：
1. 上下文中如有相關內容，必須完整、精準回覆問題，即使資料以非結構化或片段方式呈現，也應進行合理比對並擷取內容作答。
2. 若問題可透過關鍵字比對、相似語意對應從資料庫找到資訊，請進行語意理解並給出正確回答，避免錯誤回覆為找不到。
3. 僅當上下文中確實找不到任何相關資訊時，才回覆：
"不好意思未能回覆您的問題，請您簡述問題或換句話說，希望有機會為您服務。"
4. 若資料中包含 URL，請使用如下格式呈現：
有url的部分不要省略，若有url麻煩用如下格式回答，左右括弧都要用英語格式的，不要用中文格式的括號： /[content/](urllink)
其中content為內文，urllink為url連結
5. 若資料中包含換行符號或格式，請如實保留，嚴禁自動省略或合併段落。
</instruction>
<example>
    使用者問題：「我想知道如何註冊公司。」
    回答示例：「註冊公司通常需要以下步驟：1. 選擇公司名稱；2. 提交註冊申請；3. 繳納相關費用……」
</example>
<output>
直接回答答案，除了url以外沒有格式
</output>
"""

def prompt_setter(prompt):
    with open(prompt_dir + "/prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

def get_llm_response(retrieved_contexts, user_query):
    """
    使用GPT-4o-mini模型生成回應
    """
    logging.info(f"開始生成回應，上下文數量：{len(retrieved_contexts)}")
    system_prompt = ""
    try:
        system_prompt = prompt_getter()
        
        # 組建上下文提示
        context_text = "\n".join([f"{context}," for i, context in enumerate(retrieved_contexts)])
        context_text = context_text.rstrip(",\n")
        
        # 創建消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"上下文資訊：\n\n[{context_text}]\n\n用戶問題：{user_query}"}
        ]
        
        # 調用 Azure OpenAI API
        logging.info("開始調用 Azure OpenAI API")
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=0.8,
            max_tokens=512,
        )
        
        logging.info("成功生成回應")
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"生成回應時發生錯誤：{str(e)}"
        logging.error(error_msg)
        return error_msg

def rag_query(user_query):
    logging.info(f"處理RAG查詢：{user_query}")
    retrieved_contexts = queryprocess(user_query)
    
    # 使用LLM生成回應
    if len(retrieved_contexts) > 0:
        logging.info(f"找到 {len(retrieved_contexts)} 個相關上下文")
        response = get_llm_response(retrieved_contexts, user_query)
    elif hello_query(user_query):
        logging.info("檢測到問候語，回覆問候")
        response = "您好，我是我是經濟部的機器人客服「e哥」！很高興見到您！"
    else:
        logging.warning("未找到相關上下文，返回默認回覆")
        response = "不好意思未能回覆您的問題，請您簡述問題或換句話說，希望有機會為您服務！！！"
    
    logging.info("查詢處理完成")
    return response

def main():
    """
    主函數：初始化嵌入並運行互動式RAG查詢循環
    """
    logging.info("開始初始化嵌入向量")
    document_activate()
    
    logging.info("===== RAG問答系統已啟動 =====")
    logging.info("輸入您的問題，系統會檢索相關資訊並生成回答。輸入 'exit' 結束程序。")
    
    app_start_end_time = time.time()
    logging.info(f"系統啟動完成，耗時：{app_start_end_time - app_start_time:.2f} 秒")
    
    while True:
        user_query = input("\n請輸入您的問題：")
        start_time = time.time()
        
        if user_query.strip().lower() in ["exit", "quit", "退出"]:
            logging.info("使用者退出系統")
            print("感謝使用，再見！")
            break
        
        logging.info(f"開始處理查詢：{user_query}")
        response = rag_query(user_query)
        
        if response:
            print(f"\n回答：\n{response}")
            logging.info(f"查詢處理完成，耗時：{time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    logging.info("程序啟動")
    main()
    logging.info("程序結束") 