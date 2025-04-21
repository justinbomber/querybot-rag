# RAG問答系統 API 使用說明

## API概述

RAG問答系統API是一個基於本地服務的問答系統，允許用戶通過發送查詢獲取智能回答。這個系統使用了檢索增強生成（Retrieval-Augmented Generation，簡稱RAG）技術，能夠根據用戶的問題提供相關且精確的回答。

## 基本信息

- **API端點**：`http://{host}:5000/api/query`
- **HTTP方法**：POST
- **數據格式**：JSON

## 認證方法

該API使用Bearer Token認證方式。每個請求都需要在HTTP頭部包含有效的認證令牌。

- **認證方式**：Bearer Token
- **預設Token**：`ragapp-default123`

在HTTP請求頭部中添加：
```
Authorization: Bearer {YOUR_TOKEN}
```

## 請求格式

### HTTP頭部

```
Authorization: Bearer {YOUR_TOKEN}
Content-Type: application/json
Accept: application/json
```

### 請求體

請求體應為JSON格式，包含以下欄位：

```json
{
  "query": "您的問題內容"
}
```

- **query** (必填)：用戶的查詢問題，字符串類型。

## 響應格式

### 成功響應 (HTTP狀態碼 200)

```json
{
  "response": "問題的回答內容"
}
```

- **response**：系統對用戶查詢的回答。

## 錯誤處理

API可能返回以下錯誤狀態碼：

- **401 Unauthorized**：認證錯誤，Token無效或已過期。
- **403 Forbidden**：權限錯誤，提供的Token沒有查詢權限。
- **其他錯誤碼**：請參考錯誤訊息內容進行排查。

## 使用示例

### Python示例

```python
import requests
import json

# API配置
API_URL = "http://localhost:5000/api/query"
TOKEN = "{YOUR_TOKEN}"  # 請替換為您的Token

# 設定HTTP頭部
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# 準備請求數據
payload = {
    "query": "什麼是RAG系統？"
}

# 發送POST請求
try:
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # 檢查是否成功
    if response.status_code == 200:
        data = response.json()
        answer = data.get("response", "未獲得回應")
        print("回答:", answer)
    else:
        print(f"錯誤: HTTP狀態碼 {response.status_code}")
        print(f"錯誤訊息: {response.text}")
        
except Exception as e:
    print(f"發生錯誤: {e}")
```

### cURL示例

```bash
curl -X POST \
  http://localhost:5000/api/query \
  -H 'Authorization: Bearer ragapp-default123' \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -d '{
    "query": "什麼是RAG系統？"
}'
```

## 注意事項

1. 確保API服務已在本地啟動並監聽5000端口。
2. 若使用默認Token無法訪問，請聯繫系統管理員獲取有效的Token。
3. 查詢時間可能因問題複雜度不同而有所差異。
4. 如遇到連接錯誤，請確認服務器狀態。 