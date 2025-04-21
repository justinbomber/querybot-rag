# QueryBot-RAG

基於 Azure OpenAI 的檢索增強生成(RAG)問答系統，能夠根據用戶上傳的知識庫智能回答問題。系統提供直觀的Web界面和API接口，支持文檔管理、參數調整和智能問答功能。

**注意：使用時務必在./目錄下新增一個openai_token.json文件，並且在文件中填入Azure OpenAI的API金鑰**

## 功能特點

- **智能問答**：基於用戶上傳的知識庫自動檢索相關內容並生成回答
- **文檔管理**：支持上傳、刪除和切換活躍文檔
- **參數調整**：可自定義檢索數量(Top-K)和相似度閾值
- **Web界面**：提供友好的使用者交互界面
- **API接口**：支持通過API進行問答，便於系統集成
- **權限控制**：基於Token的API訪問控制
- **容器化部署**：支持Docker容器化部署

## 技術架構

- **後端**：Flask, LlamaIndex, Azure OpenAI
- **前端**：HTML, CSS, JavaScript, Bootstrap, jQuery
- **嵌入模型**：Azure OpenAI Embeddings
- **生成模型**：Azure OpenAI GPT模型
- **部署**：Docker

## 安裝與部署

### 環境要求

- Docker
- 或 Python 3.12+

### Docker部署

1. 拉取Docker映像
   ```bash
   docker pull justinbomber/azure-openai-embedding:1.0.0
   ```

2. 運行容器
   ```bash
   docker run --rm -p 5000:5000 justinbomber/azure-openai-embedding:1.0.0
   ```

3. 在瀏覽器中訪問
   ```
   http://localhost:5000
   ```

### 本地運行

1. 克隆專案
   ```bash
   git clone <repository-url>
   cd querybot-rag
   ```

2. 安裝依賴
   ```bash
   pip install -r requirements.txt
   ```

3. 配置OpenAI API
   更新`openai_token.json`文件中的API設置

4. 運行應用
   ```bash
   python app.py
   ```

## 使用說明

### 1. 上傳知識文件

- 點擊"上傳文件"卡片中的"選擇文件"按鈕
- 選擇一個.txt格式的文件
- 點擊"上傳"按鈕

### 2. 設置活躍文件

- 在"文件管理"卡片中選擇一個已上傳的文件
- 點擊"設為啟用文件"按鈕

### 3. 調整參數

- 在"參數設置"卡片中調整Top K值和相似度閾值
- 點擊"保存設置"按鈕

### 4. 進行問答

- 在輸入框中輸入問題
- 點擊"發送"按鈕或按Enter鍵
- 系統會在對話框中顯示回答

## API文檔

### 認證方法

- **認證方式**：Bearer Token
- **預設Token**：`ragapp-default123`

### 主要API

#### 問答API

- **端點**：`/api/query`
- **方法**：POST
- **請求格式**：
  ```json
  {
    "query": "您的問題內容"
  }
  ```
- **回應格式**：
  ```json
  {
    "response": "問題的回答內容"
  }
  ```

#### 文件管理API

- **列出文件**：`/api/files` (GET)
- **上傳文件**：`/api/upload` (POST)
- **設置活躍文件**：`/api/files/active` (POST)
- **刪除文件**：`/api/files/delete` (POST)

#### 參數設置API

- **獲取設置**：`/api/settings` (GET)
- **設置Top-K**：`/api/settings/top_k` (POST)
- **設置閾值**：`/api/settings/threshold` (POST)

*詳細API文檔請參考`api_usage.md`*

## 檔案結構

```
├── app.py              # Flask應用主文件
├── main.py             # RAG功能核心實現
├── embedding.py        # 向量嵌入和檢索功能
├── chatbot.py          # 聊天機器人功能
├── requirements.txt    # 依賴包列表
├── dockerfile          # Docker配置文件
├── templates/          # HTML模板
├── static/             # 靜態資源
├── data/               # 用戶上傳的知識文件
├── embedding_store/    # 嵌入向量存儲
└── logs/               # 日誌文件
```

## 貢獻與維護

### 報告問題

如果您在使用過程中遇到問題，請通過以下方式報告：
1. 在項目的Issue頁面提交問題報告
2. 描述清楚問題的重現步驟和預期行為

### 貢獻代碼

1. Fork本專案
2. 創建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟一個Pull Request

## 授權協議

本專案採用 [MIT 授權協議](LICENSE) 