import requests
import time
import json
import os
import sys

# RAG問答系統API設定
API_URL = "http://localhost:5000/api/query"  # 本地Flask應用的API端點
DEFAULT_TOKEN = "ragapp-default123"  # 預設管理員Token

def print_title():
    """列印標題和說明"""
    print("=" * 50)
    print("RAG問答系統 API 測試工具")
    print("=" * 50)
    print("此測試工具用於與RAG問答系統API進行交互")
    print("命令:")
    print("  exit/quit: 退出程序")
    print("  token: 顯示當前使用的Token")
    print("  change_token: 更換Token")
    print("=" * 50)

def get_token():
    """取得用戶輸入的Token或使用預設Token"""
    print("請輸入API Token（直接按Enter使用預設Token）：")
    token = input().strip()
    if not token:
        print(f"使用預設Token: {DEFAULT_TOKEN}")
        return DEFAULT_TOKEN
    else:
        return token

def main():
    print_title()
    
    # 取得Token
    token = get_token()
    
    # 設定HTTP頭部
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    while True:
        # 取得用戶輸入
        user_input = input("\n請輸入問題 (輸入 'exit' 或 'quit' 退出，'token' 查看當前Token，'change_token' 更換Token)：")
        start_time = time.time()

        
        # 檢查特殊命令
        if user_input.lower() in ["exit", "quit"]:
            print("感謝使用，再見！")
            break
        elif user_input.lower() == "token":
            print(f"當前使用的Token: {token}")
            continue
        elif user_input.lower() == "change_token":
            token = get_token()
            headers["Authorization"] = f"Bearer {token}"
            continue
        
        # 準備請求資料
        payload = {
            "query": user_input
        }
        
        try:
            # 發送POST請求
            response = requests.post(API_URL, headers=headers, json=payload)
            
            # 檢查是否成功
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "未獲得回應")
                end_time = time.time()
                duration = end_time - start_time
                
                # 輸出結果
                print("=" * 50)
                print(f"查詢時間: {duration:.2f} 秒")
                print("-" * 50)
                print("回答:", answer)
                print("-" * 50)
                print("=" * 50)
            elif response.status_code == 401:
                print("認證錯誤: Token無效或已過期")
                print(response.json())
                retry = input(f"是否要更換Token? 目前token為:{token} (y/n): ")
                if retry.lower() == "y":
                    token = get_token()
                    headers["Authorization"] = f"Bearer {token}"
            elif response.status_code == 403:
                print("權限錯誤: 您的Token沒有查詢權限")
                retry = input("是否要更換Token? (y/n): ")
                if retry.lower() == "y":
                    token = get_token()
                    headers["Authorization"] = f"Bearer {token}"
            else:
                print(f"錯誤: HTTP狀態碼 {response.status_code}")
                print(f"錯誤訊息: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("連接錯誤: 無法連接到API服務器。請確認服務器已啟動且URL正確。")
            retry = input("是否要重試? (y/n): ")
            if retry.lower() != "y":
                break
        except Exception as e:
            print(f"發生錯誤: {e}")
            print("回應內容:", response.text if 'response' in locals() else "無回應")

if __name__ == "__main__":
    main()