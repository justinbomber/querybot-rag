from flask import Flask, request, jsonify, render_template, send_from_directory, g
import os
import random
import string
import time
import json
from functools import wraps
from werkzeug.utils import secure_filename
import sys
sys.path.append('.')
from main import rag_query
from embedding import (
    top_k_getter, top_k_setter, 
    score_threshold_getter, score_threshold_setter,
    active_file, activate_file_getter, 
    document_activate, document_embedding,
    document_delete, document_list,
    hello_init, hello_query
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './data'  # 因為現在app.py在local_llm目錄下，所以相對路徑是正確的
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上傳文件大小為16MB
app.config['TOKEN_FILE'] = './tokens.json'  # Token存儲文件

# Token格式: "ragapp-{12位中英隨機數}"
DEFAULT_TOKEN = "ragapp-default123"  # 預設token，具有所有權限

# 全局變數存儲處理進度
embedding_progress = 0
embedding_file = None
embedding_in_progress = False

# Token儲存結構
tokens = {
    DEFAULT_TOKEN: {
        "permissions": ["all"],  # 特殊權限"all"表示可以訪問所有API
        "created_at": time.time(),
        "last_used": time.time(),
        "description": "預設管理員Token"
    }
}

# 載入已存在的token
def load_tokens():
    global tokens
    if os.path.exists(app.config['TOKEN_FILE']):
        try:
            with open(app.config['TOKEN_FILE'], 'r', encoding='utf-8') as f:
                loaded_tokens = json.load(f)
                # 確保默認token存在
                if DEFAULT_TOKEN not in loaded_tokens:
                    loaded_tokens[DEFAULT_TOKEN] = tokens[DEFAULT_TOKEN]
                tokens = loaded_tokens
        except Exception as e:
            print(f"載入tokens時出錯: {e}")
    else:
        save_tokens()  # 如果文件不存在，保存默認token

# 保存tokens到文件
def save_tokens():
    try:
        with open(app.config['TOKEN_FILE'], 'w', encoding='utf-8') as f:
            json.dump(tokens, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"保存tokens時出錯: {e}")

# 生成隨機token
def generate_token():
    random_part = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    return f"ragapp-{random_part}"

# Token認證裝飾器
def token_required(endpoint=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            load_tokens()
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                print(f"缺少認證Token: {auth_header}")
                return jsonify({'error': '缺少認證Token'}), 401
            
            token = auth_header.split('Bearer ')[1].strip()
            if token not in tokens:
                print(f"無效的Token: {token}")
                return jsonify({'error': '無效的Token'}), 401
            
            # 檢查token權限
            permissions = tokens[token]["permissions"]
            if "all" not in permissions and endpoint not in permissions:
                return jsonify({'error': '權限不足'}), 403
            
            # 更新最後使用時間
            tokens[token]["last_used"] = time.time()
            save_tokens()
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# 確保上傳文件夾存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化tokens，啟用當前文件
load_tokens()
document_activate()  # 啟用現有文件
hello_init()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
@token_required('query')
def query():
    data = request.json
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({'error': '查詢不能為空'}), 400
    try:
        response = rag_query(user_query)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
@token_required('upload')
def upload():
    global embedding_progress, embedding_file, embedding_in_progress
    
    if 'file' not in request.files:
        return jsonify({'error': '沒有文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未選擇文件'}), 400
    if file and file.filename.endswith('.txt'):
        # 自定義安全文件名處理，保留中文字符
        filename = file.filename
        # 替換可能有問題的字符
        filename = filename.replace('/', '_').replace('\\', '_').replace('..', '_')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 清理當前的進度狀態
        embedding_progress = 0
        embedding_file = filepath
        embedding_in_progress = True
        
        # 啟動非阻塞處理
        import threading
        
        def process_file():
            global embedding_progress, embedding_in_progress
            try:
                # 激活文件
                # active_file(filepath)
                
                # 開始處理嵌入向量，這會生成進度
                for progress in document_embedding(filepath):
                    embedding_progress = progress
                
                # 處理完成
                embedding_progress = 100
                embedding_in_progress = False
            except Exception as e:
                print(f"處理文件嵌入時出錯: {e}")
                embedding_in_progress = False
        
        # 啟動處理線程
        thread = threading.Thread(target=process_file)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'message': '文件上傳成功，開始處理嵌入向量'
        })
    else:
        return jsonify({'error': '僅支持.txt文件'}), 400

@app.route('/api/upload/progress', methods=['GET'])
@token_required('upload')
def get_upload_progress():
    """獲取文件處理進度"""
    global embedding_progress, embedding_file, embedding_in_progress
    
    return jsonify({
        'progress': embedding_progress,
        'file': os.path.basename(embedding_file) if embedding_file else None,
        'in_progress': embedding_in_progress
    })

@app.route('/api/settings', methods=['GET'])
@token_required('settings')
def get_settings():
    try:
        return jsonify({
            'top_k': top_k_getter(),
            'score_threshold': score_threshold_getter(),
            'active_file': activate_file_getter()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/top_k', methods=['POST'])
@token_required('settings')
def update_top_k():
    data = request.json
    top_k = data.get('top_k')
    if top_k is None:
        return jsonify({'error': '缺少top_k參數'}), 400
    try:
        top_k_setter(int(top_k))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/threshold', methods=['POST'])
@token_required('settings')
def update_threshold():
    data = request.json
    threshold = data.get('threshold')
    if threshold is None:
        return jsonify({'error': '缺少threshold參數'}), 400
    try:
        score_threshold_setter(float(threshold))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files', methods=['GET'])
@token_required('files')
def get_files():
    try:
        files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                if f.endswith('.txt')]
        active = activate_file_getter()
        return jsonify({
            'files': files,
            'active_file': active
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/active', methods=['POST'])
@token_required('files')
def set_active_file():
    data = request.json
    file_path = data.get('file_path')
    if not file_path:
        return jsonify({'error': '缺少file_path參數'}), 400
    try:
        result = active_file(file_path)
        if result.startswith('200'):
            return jsonify({'success': True})
        else:
            return jsonify({'error': result}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Token管理API

@app.route('/api/token/generate', methods=['POST'])
@token_required('token_manage')
def generate_api_token():
    data = request.json
    description = data.get('description', '查詢Token')
    
    # 生成新token
    new_token = generate_token()
    tokens[new_token] = {
        "permissions": ["query"],  # 只有查詢權限
        "created_at": time.time(),
        "last_used": time.time(),
        "description": description
    }
    save_tokens()
    
    return jsonify({
        'success': True,
        'token': new_token,
        'permissions': ["query"],
        'description': description
    })

@app.route('/api/token/list', methods=['GET'])
@token_required('token_manage')
def list_tokens():
    token_list = []
    for token, details in tokens.items():
        if token != DEFAULT_TOKEN:  # 不顯示默認token
            token_list.append({
                'token': token,
                'permissions': details['permissions'],
                'created_at': details['created_at'],
                'last_used': details['last_used'],
                'description': details.get('description', '')
            })
    return jsonify({'tokens': token_list})

@app.route('/api/token/revoke', methods=['POST'])
@token_required('token_manage')
def revoke_token():
    data = request.json
    token = data.get('token')
    if not token:
        return jsonify({'error': '缺少token參數'}), 400
    
    if token == DEFAULT_TOKEN:
        return jsonify({'error': '不能撤銷默認Token'}), 400
    
    if token in tokens:
        del tokens[token]
        save_tokens()
        return jsonify({'success': True})
    else:
        return jsonify({'error': '找不到指定的Token'}), 404

@app.route('/api/files/list', methods=['GET'])
@token_required('files')
def list_files():
    """獲取嵌入存儲中的所有文件列表"""
    try:
        # 獲取嵌入存儲中的文件列表
        embedding_files = document_list()
        # 獲取data目錄中的文件列表
        data_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                if f.endswith('.txt')]
        active = activate_file_getter()
        return jsonify({
            'embedding_files': embedding_files,
            'data_files': data_files,
            'active_file': active
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/delete', methods=['POST'])
@token_required('files')
def delete_file():
    """刪除文件及其嵌入數據"""
    data = request.json
    file_path = data.get('file_path')
    if not file_path:
        return jsonify({'error': '缺少file_path參數'}), 400
    try:
        result = document_delete(file_path)
        if result:
            return jsonify({'success': True})
        else:
            return jsonify({'error': '刪除失敗，可能是文件正在使用中'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 使用Werkzeug內建的多工作進程支持
    # 在Windows上，這是比較好的多進程選項
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, processes=4) 