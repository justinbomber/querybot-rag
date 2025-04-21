# RAG問答系統

這是一個基於RAG (Retrieval Augmented Generation) 的問答系統，使用Azure OpenAI提供的API服務。系統包括直觀的Web界面，允許用戶上傳知識文件、調整參數並進行問答互動。

## 功能特點

- 智能問答：基於用戶上傳的知識庫回答問題
- 知識管理：上傳和選擇知識文件
- 參數調整：設置相似度檢索的數量和閾值
- 直觀界面：美觀易用的Web交互界面

## 技術棧

- 後端：Flask, LlamaIndex, Azure OpenAI
- 前端：HTML, CSS, JavaScript, Bootstrap, jQuery
- 嵌入模型：Azure OpenAI Embeddings
- 生成模型：Azure OpenAI GPT模型

## 安裝與運行

### 環境要求

docker

### 安裝步驟

1. pull docker image
   ```
   docker pull justinbomber/azure-openai-embedding:1.0.0
   ```

2. 運行docker image
   ```
   docker run --rm -p 5000:5000 justinbomber/azure-openai-embedding:1.0.0
   ```

3. 在瀏覽器中訪問
   ```
   http://localhost:5000
   ```

## 使用說明

1. **上傳知識文件**
   - 點擊"上傳文件"卡片中的"選擇文件"按鈕
   - 選擇一個.txt格式的文件
   - 點擊"上傳"按鈕

2. **設置活躍文件**
   - 在"文件管理"卡片中選擇一個已上傳的文件
   - 點擊"設為啟用文件"按鈕

3. **調整參數**
   - 在"參數設置"卡片中調整Top K值和相似度閾值
   - 點擊"保存設置"按鈕

4. **進行問答**
   - 在輸入框中輸入問題
   - 點擊"發送"按鈕或按Enter鍵
   - 系統會在對話框中顯示回答

## 文件結構

- `app.py`: Flask應用主文件
- `local_llm/`: 核心功能模塊
  - `embedding.py`: 向量嵌入和檢索相關功能
  - `main.py`: RAG查詢和回應生成
- `static/`: 靜態資源
  - `css/`: 樣式表
  - `js/`: JavaScript文件
- `templates/`: HTML模板
- `requirements.txt`: 依賴包列表 