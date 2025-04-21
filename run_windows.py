from waitress import serve
from app import app
import os
import sys

if __name__ == '__main__':
    # 設置工作線程數，默認為4
    threads = int(os.environ.get('WAITRESS_THREADS', 8))
    print(f"啟動Waitress服務器，使用 {threads} 個線程...")
    
    # 啟動服務
    serve(app, host='0.0.0.0', port=5000, threads=threads) 