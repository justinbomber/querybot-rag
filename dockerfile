# 使用官方 Python 映像
FROM python:3.12-slim

# 設定工作目錄
WORKDIR /app

# 複製需求檔案並安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼
COPY . /app/

# 建立資料夾以儲存嵌入向量
RUN mkdir -p /app/embedding_store

# 暴露5000端口
EXPOSE 5000

# 設定執行命令 - 使用gunicorn運行app.py，啟用多工作進程和線程
CMD ["gunicorn", "--workers=4", "--threads=2", "--timeout=120", "-b", "0.0.0.0:5000", "app:app"]