import os
import re
import shutil
import time
import numpy as np
import math  # 保留這個import
import json  # 添加json模塊
import logging
from datetime import datetime
from transformers import GPT2Tokenizer
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core.indices.loading import load_index_from_storage

# 創建logs目錄（如果不存在）
if not os.path.exists('logs'):
    os.makedirs('logs')

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/embedding.log', encoding='utf-8'),
        logging.StreamHandler()  # 同時輸出到控制台
    ]
)

activate_file = None
try:
    with open("./setting.json", "r", encoding="utf-8") as f:
        setting = json.load(f)
        activate_file = setting["activate_file"]
except Exception as e:
    activate_file = "./data/常見問答_result.txt"
    logging.error(f"讀取setting.json時發生錯誤：{e}，使用預設檔案：{activate_file}")

# 指定 persist 資料夾
root_dir = "./embedding_store"
dirname, basename = os.path.split(activate_file)
active_file_name, ext = os.path.splitext(basename)
persist_dir = os.path.join(root_dir, active_file_name)

load_from_disk = False
document_embedding_bool = False
chunks = []
normalized_embeddings = []
document_embedding_norm = None
document_embedding = None
normalized_embeddings = None

hello_document_embedding_norm = None
hello_document_embedding = None
hello_normalized_embeddings = None
hello_chunks = None


if os.path.exists("./setting.json"):
    with open("./setting.json", "r", encoding="utf-8") as f:
        setting = json.load(f)
    top_k = setting["top_k"]
    score_threshold = setting["score_threshold"]
    logging.info(f"從配置文件讀取設置：top_k={top_k}, score_threshold={score_threshold}")
else:
    top_k = 3
    score_threshold = 0.0
    logging.warning("未找到配置文件，使用默認參數：top_k=3, score_threshold=0.0")

def normalize(vector):
    """
    正規化向量
    
    將輸入向量正規化為單位向量（長度為1）
    
    @param vector: 要正規化的向量
    @return: 正規化後的向量
    """
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

# 初始化 Azure OpenAI 嵌入模型 (請確認參數皆正確)
embed_model = AzureOpenAIEmbedding(
    model="xxx",
    deployment_name="xxx",
    api_key="xxxxx",
    azure_endpoint="https://xxxx.openai.azure.com/",
    api_version="xxx",
)

def upload_file(input_file_path):
    """
    上傳檔案到指定目錄
    
    目前為待實現的函數
    
    @param input_file_path: 要上傳的檔案路徑
    @return: None
    """
    # TODO: 上傳檔案到指定目錄
    pass

def top_k_setter(top_k):
    logging.info(f"設置 top_k 參數為：{top_k}")
    with open("./setting.json", "r", encoding="utf-8") as f:
        settings = json.load(f)
        settings["top_k"] = top_k
    with open("./setting.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)
    logging.info(f"成功更新 top_k 參數為：{top_k}")

def score_threshold_setter(score_threshold):
    logging.info(f"設置 score_threshold 參數為：{score_threshold}")
    with open("./setting.json", "r", encoding="utf-8") as f:
        settings = json.load(f)
        settings["score_threshold"] = score_threshold
    with open("./setting.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)
    logging.info(f"成功更新 score_threshold 參數為：{score_threshold}")

def active_file(file_path):
    """
    設定啟用檔案
    
    將指定的文本檔案設定為啟用檔案，用於嵌入向量計算
    
    @param file_path: 要啟用的檔案路徑
    @return: None
    """
    global activate_file, persist_dir, root_dir
    logging.info(f"開始設定啟用檔案：{file_path}")
    dirname, basename = os.path.split(file_path)
    active_file_name, ext = os.path.splitext(basename)
    if ext == ".txt":
        thepersist_dir = os.path.join(root_dir, active_file_name)
        setting = {}
        # 如果setting.json不存在，則創建，存在則更改activate_file
        with open("./setting.json", "r", encoding="utf-8") as f:
            settings = json.load(f)
        # 修改指定欄位
        settings["activate_file"] = file_path
        # 寫回檔案（只改動這個欄位，保留其他內容）
        with open("./setting.json", "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=4)
        # 設定指定檔案為啟用檔案
        with open("./setting.json", "r", encoding="utf-8") as f:
            setting = json.load(f)
        activate_file = setting["activate_file"]
        document_activate(thepersist_dir)
        logging.info(f"已設定啟用檔案：{activate_file}")
        return "200: active file success"
    else:
        logging.error("不支援的檔案類型")
        return "400: active file failed"

def top_k_getter():
    with open("./setting.json", "r", encoding="utf-8") as f:
        settings = json.load(f)
        return settings["top_k"]

def score_threshold_getter():
    with open("./setting.json", "r", encoding="utf-8") as f:
        settings = json.load(f)
        return settings["score_threshold"]

def activate_file_getter():
    with open("./setting.json", "r", encoding="utf-8") as f:
        settings = json.load(f)
        return settings["activate_file"]

def document_activate(persist_dir=persist_dir):
    """
    初始化嵌入向量
    
    初始化或載入已存在的文本嵌入向量。如果存在持久化的嵌入向量，則直接載入；
    否則，從原始文本計算嵌入向量並存儲。
    
    @param persist_dir: 持久化目錄路徑，默認為None
    @return: None
    """
    global load_from_disk, \
        document_embedding_bool, \
        document_embedding_norm, \
        document_embedding, \
        normalized_embeddings, \
        chunks, \
        activate_file
    
    logging.info(f"開始初始化嵌入向量，目錄：{persist_dir}")
    #載入既有向量
    document_embedding_bool = False
    load_from_disk = False

    # mkdir if not exist
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
        logging.info(f"創建持久化目錄：{persist_dir}")

    if os.path.exists(persist_dir + "/document_embedding_norm.npy") and \
        os.path.exists(persist_dir + "/document_embedding.npy") and \
        os.path.exists(persist_dir + "/normalized_embeddings.npy") and \
        os.path.exists(persist_dir + "/chunks.json"):
        document_embedding_bool = True

    if os.path.exists(persist_dir):
    # 檢查目錄下是否有檔案（根據實際情況可更嚴格檢查特定檔案，如 docstore.json、default__vector_store.json 等）
        if any(os.scandir(persist_dir)):
            load_from_disk = True

    if document_embedding_bool:
        logging.info("已經計算過節點嵌入向量，直接載入索引...")
        document_embedding_norm = np.load(persist_dir + "/document_embedding_norm.npy")
        document_embedding = np.load(persist_dir + "/document_embedding.npy")
        normalized_embeddings = np.load(persist_dir + "/normalized_embeddings.npy")
        
        # 載入chunks
        try:
            with open(persist_dir + "/chunks.json", "r", encoding="utf-8") as f:
                chunks = json.load(f)
            logging.info(f"成功載入 {len(chunks)} 個文本段落")
        except FileNotFoundError:
            logging.warning("找不到chunks.json文件，某些功能可能無法正常運作")
    elif load_from_disk:
        # 依據官方文件載入持久化的索引
        logging.info("發現持久化資料，從磁碟中載入索引...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        try:
            # 不指定index_id，自動載入唯一索引，關鍵是要傳入embed_model參數
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            logging.info("成功載入索引")
            
            # 從docstore中獲取所有節點
            nodes = list(index.docstore.docs.values())
            logging.info(f"成功載入 {len(nodes)} 個節點")
            
            # 獲取每個節點的文本內容
            chunks = [node.text for node in nodes]
            
            # 直接計算嵌入向量，SimpleVectorStore不支持直接獲取節點嵌入
            logging.info("計算節點嵌入向量...")
            embeddings = []
            for node in nodes:
                try:
                    embedding_response = embed_model(nodes=[node])
                    emb = embedding_response[0].embedding
                    embeddings.append(emb)
                except Exception as e:
                    logging.error(f"計算節點嵌入時發生錯誤：{e}")
            
            if embeddings:
                normalized_embeddings = [normalize(emb) for emb in embeddings]
                logging.info(f"已計算 {len(normalized_embeddings)} 個嵌入向量")
                np.save(persist_dir + "/normalized_embeddings.npy", normalized_embeddings)
                
                # 保存chunks
                with open(persist_dir + "/chunks.json", "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False)
                logging.info(f"已保存 {len(chunks)} 個文本段落")
                
                # 計算整個文件的代表向量
                document_embedding = np.mean(embeddings, axis=0)
                document_embedding_norm = normalize(document_embedding)
                # save document_embedding_norm and document_embedding to file
                np.save(persist_dir + "/document_embedding_norm.npy", document_embedding_norm)
                np.save(persist_dir + "/document_embedding.npy", document_embedding)
                
                logging.info("已計算文檔平均嵌入向量")
            else:
                logging.error("無法計算任何嵌入向量，請檢查處理流程。")
                load_from_disk = False
                
            logging.info("索引讀取成功。")
        except Exception as e:
            logging.error(f"載入索引失敗，將重新計算嵌入。錯誤訊息：{e}")
            load_from_disk = False
    
    logging.info("初始化嵌入向量完成")

def document_embedding(input_file_path):
    """
    計算文件的嵌入向量
    
    計算文件的嵌入向量，並返回嵌入向量
    
    @param persist_dir: 持久化目錄路徑，默認為None
    @return: 嵌入向量
    """
    global root_dir
    logging.info(f"開始計算文件嵌入向量：{input_file_path}")
    input_dir, input_basename = os.path.split(input_file_path)
    input_file_name, input_file_ext = os.path.splitext(input_basename)
    local_persist_dir = os.path.join(root_dir, input_file_name)
    if not os.path.exists(local_persist_dir):
        os.makedirs(local_persist_dir)
        logging.info(f"創建持久化目錄：{local_persist_dir}")

    def custom_split_by_line(text: str, max_tokens: int = 500):
        """
        自訂分割文本的函數
        
        將文本按行分割，並確保每個分割後的片段不超過指定的token數量
        
        @param text: 要分割的文本
        @param max_tokens: 每個片段的最大token數量，默認為500
        @return: 分割後的文本片段列表
        """
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        chunks = []
        paragraphs = text.split("\n")
        for para in paragraphs:
            if (len(para) < 20):
                logging.debug(f"段落長度不足20字：{para}")
            tokens = tokenizer.tokenize(para)
            if len(tokens) <= max_tokens:
                chunks.append(para)
            else:
                # 長段內再細切
                token_chunks = [
                    tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)
                ]
                for token_chunk in token_chunks:
                    chunk_text = tokenizer.convert_tokens_to_string(token_chunk)
                    chunks.append(chunk_text)
        return chunks

    # def custom_split_by_clause(text: str, max_tokens: int = 500):
    #     """
    #     自訂分割文本的函數

    #     將文本按「第***條」標題分割，並確保每個分割後的片段不超過指定的 token 數量

    #     @param text: 要分割的文本
    #     @param max_tokens: 每個片段的最大 token 數量，默認為 500
    #     @return: 分割後的文本片段列表
    #     """
    #     # 初始化 tokenizer
    #     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    #     # 使用正則表達式切分，保留「第***條」作為分割點
    #     pattern = r'(第\d+條)'
    #     parts = re.split(pattern, text)

    #     chunks = []
    #     # 按照標題與內容配對
    #     for i in range(1, len(parts), 2):
    #         header = parts[i]
    #         body = parts[i+1] if i+1 < len(parts) else ''
    #         segment = header + body

    #         tokens = tokenizer.tokenize(segment)
    #         if len(tokens) <= max_tokens:
    #             chunks.append(segment)
    #         else:
    #             # 若超過 token 限制，則按 token 數再次細分
    #             token_chunks = [
    #                 tokens[j:j + max_tokens] for j in range(0, len(tokens), max_tokens)
    #             ]
    #             for token_chunk in token_chunks:
    #                 chunk_text = tokenizer.convert_tokens_to_string(token_chunk)
    #                 chunks.append(chunk_text)

    #     return chunks


    def process_text_to_nodes(chunks_list):
        """
        將文本片段處理成節點
        
        根據提供的文本片段列表創建TextNode節點
        
        @param chunks_list: 文本片段列表
        @return: TextNode節點列表
        """
        nodes = []
        for chunk in chunks_list:
            node = TextNode(text=chunk)
            nodes.append(node)
        return nodes

    # 若無既有向量，則讀取原始文本並計算嵌入
    try:
        embedding_progress = 0
        # 若無持久化資料，則讀取原始文本並計算嵌入
        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                text_data = f.read()
            logging.info(f"成功讀取文件：{input_file_path}")
        except FileNotFoundError:
            # 嘗試另一種相對路徑
            logging.warning(f"嘗試另一種相對路徑：{input_file_path}")
            try:
                with open(input_file_path, "r", encoding="utf-8") as f:
                    text_data = f.read()
                logging.info(f"成功讀取文件：{activate_file}")
            except FileNotFoundError:
                logging.error(f"找不到檔案。請確認檔案位置是否正確。")
                exit(1)

        logging.info("開始處理原始文本...")
        # 使用適合中文的分詞器切分文本
        chunks = custom_split_by_line(text_data, max_tokens=1000)
        # chunks = custom_split_by_clause(text_data, max_tokens=1000)
        
        # 使用文本片段列表創建節點
        nodes = process_text_to_nodes(chunks)
        
        logging.info(f"切割後共取得 {len(chunks)} 段")
        for i in range(min(3, len(chunks))):
            preview = chunks[i].replace('\n', ' ')
            logging.debug(f"段落 {i+1} 預覽: '{preview}...'")

        embeddings = []
        for i, node in enumerate(nodes):
            try:
                if i+2 == len(chunks):
                    break
                embedding_response = embed_model(nodes=[node])
                emb = embedding_response[0].embedding
                embeddings.append(emb)
                logging.debug(f"第 {i+1} 段嵌入計算完成；向量長度：{len(emb)}")
                if i < 3:
                    logging.debug(f"向量前5個數值: {emb[:5]}")
                embedding_progress = (i/len(chunks))*100
                if embedding_progress > 1:
                    yield embedding_progress-1
                else:
                    yield embedding_progress
            except Exception as e:
                logging.error(f"計算第 {i+1} 段嵌入時發生錯誤：{e}")

        if len(embeddings) < 1:
            logging.error("無法獲取任何嵌入向量，請檢查處理流程。")
            exit(1)

        normalized_embeddings = [normalize(emb) for emb in embeddings]
        np.save(local_persist_dir + "/normalized_embeddings.npy", normalized_embeddings)

        # 保存chunks
        with open(local_persist_dir + "/chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        logging.info(f"已保存 {len(chunks)} 個文本段落")

        # 利用所有節點及嵌入模型建立向量索引
        try:
            index = VectorStoreIndex(nodes, embed_model=embed_model)
            # 儲存索引至磁碟
            index.storage_context.persist(persist_dir=local_persist_dir)
            # 保存索引ID供參考
            with open(os.path.join(local_persist_dir, "index_id.txt"), "w") as f:
                f.write(index.index_id)
            logging.info(f"索引已儲存至：{local_persist_dir}，索引ID：{index.index_id}")
        except Exception as e:
            logging.error(f"建立索引時發生錯誤：{e}")
            logging.warning("僅使用嵌入向量進行相似度比較。")

        # 計算整個文件的代表向量（取所有段落嵌入平均值，再正規化）
        np.save(local_persist_dir + "/document_embedding.npy", document_embedding)
        np.save(local_persist_dir + "/document_embedding_norm.npy", document_embedding_norm)
        logging.info("所有段落嵌入計算完成，已計算平均嵌入向量並正規化。")
        logging.debug(f"平均向量前10個數值: {document_embedding[:10]}")
        logging.debug(f"正規化後的平均向量前10個數值: {document_embedding_norm[:10]}")
        embedding_progress = 100
        yield embedding_progress

    except Exception as e:
        logging.error(f"處理原始文本時發生錯誤：{e}")
        exit(1)

def document_delete(file_path):
    """
    刪除持久化資料
    """
    global root_dir
    logging.info(f"嘗試刪除文件：{file_path}")
    del_dirname, del_file = os.path.split(file_path)
    act_dirname, act_file = os.path.split(activate_file)
    if del_file == act_file:
        logging.error(f"文件 {file_path} 正在使用中，無法刪除")
        return False
    else:
        try:
            del_file_name, del_file_ext = os.path.splitext(del_file)
            os.remove(file_path)
            shutil.rmtree(os.path.join(root_dir, del_file_name))
            logging.info(f"成功刪除文件：{file_path}")
            return True
        except Exception as e:
            logging.error(f"刪除文件失敗，錯誤：{e}")
            return False

def document_list():
    """
    列出所有持久化資料
    """
    global root_dir
    logging.info(f"列出目錄 {root_dir} 中的所有文件")
    return os.listdir(root_dir)

def queryprocess(query):
    """
    處理查詢
    
    處理用戶輸入的查詢，計算查詢與文本片段的相似度，返回最相關的片段
    
    @param query: 用戶輸入的查詢文本
    @return: 包含相似度和內容的結果列表，或狀態碼字符串
    """
    global document_embedding_norm, \
           document_embedding, \
           normalized_embeddings, \
           chunks, \
           top_k, \
           score_threshold

    logging.info(f"處理查詢：{query}")
    output = []
    # 原有的餘弦相似度函數（用於字典類型向量）
    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        numerator = np.dot(vec1, vec2)
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denominator == 0:
            return np.array([0.0], dtype=np.float32)
        return np.array([numerator / denominator], dtype=np.float32)

    # 新增的餘弦相似度函數（用於numpy數組）
    def cosine_similarity_array(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    start_time = time.monotonic_ns()

    with open("setting.json", "r", encoding="utf-8") as f:
        setting = json.load(f)
        score_threshold = setting["score_threshold"]
        top_k = setting["top_k"]
    
    logging.debug(f"使用參數：top_k={top_k}, score_threshold={score_threshold}")

    # 將查詢文本封裝成 TextNode 並計算嵌入向量
    query_node = TextNode(text=query)
    try:
        query_embedding_response = embed_model(nodes=[query_node])
    except Exception as e:
        logging.error(f"查詢嵌入計算失敗：{e}")
        return "400: query embedding calculation failed"
    query_embedding = query_embedding_response[0].embedding
    query_embedding_norm = normalize(query_embedding)

    # 使用正規化後的平均向量進行查詢相似度計算
    if document_embedding_norm is not None:
        similarity = cosine_similarity(query_embedding_norm, document_embedding_norm)
        logging.info(f"查詢與整個文件相關性 (餘弦相似度)：{similarity[0]:.4f}")
    
    # 修正條件判斷
    if normalized_embeddings is not None and len(normalized_embeddings) > 0:
        chunk_similarities = []
        for i, emb_norm in enumerate(normalized_embeddings):
            sim = cosine_similarity(query_embedding_norm, emb_norm)
            if sim[0] >= score_threshold:
                chunk_similarities.append((i, sim[0]))
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        # 轉換為毫秒
        end_time = time.monotonic_ns()
        logging.info(f"與查詢最相關的{top_k}個段落，查詢時間：{(end_time - start_time) / 1000000:.2f}ms")
        for i in range(min(top_k, len(chunk_similarities))):
            chunk_idx, sim = chunk_similarities[i]
            if chunk_idx < len(chunks):
                preview = chunks[chunk_idx].replace('\n', ' ')
                output.append({
                    "similarity": sim,
                    "content": preview
                })
                # logging.debug記錄移除，不必要的輸出
            else:
                logging.warning(f"段落索引 {chunk_idx} 超出範圍，最大索引為 {len(chunks)-1}")
    else:
        logging.warning("無可用的節點嵌入向量或chunks尚未載入")
    
    logging.info(f"查詢處理完成，找到 {len(output)} 個相關片段")
    return output

def hello_init():
    global hello_document_embedding_norm, hello_document_embedding, hello_normalized_embeddings, hello_chunks
    logging.info("初始化問候語辨識系統")
    hello_dir = "./hellotext"
    hello_document_embedding_norm = np.load(os.path.join(hello_dir, "document_embedding_norm.npy"))
    hello_document_embedding = np.load(os.path.join(hello_dir, "document_embedding.npy"))
    hello_normalized_embeddings = np.load(os.path.join(hello_dir, "normalized_embeddings.npy"))
    with open(os.path.join(hello_dir, "chunks.json"), "r", encoding="utf-8") as f:
        hello_chunks = json.load(f)
    logging.info("問候語辨識系統初始化完成")
    return True

def hello_query(query):
    global hello_document_embedding_norm, hello_document_embedding, hello_normalized_embeddings, hello_chunks
    logging.debug(f"檢查問候語：{query}")
    output = []
    # 原有的餘弦相似度函數（用於字典類型向量）
    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        numerator = np.dot(vec1, vec2)
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denominator == 0:
            return np.array([0.0], dtype=np.float32)
        return np.array([numerator / denominator], dtype=np.float32)

    # 使用正規化後的平均向量進行查詢相似度計算
    query_node = TextNode(text=query)
    try:
        query_embedding_response = embed_model(nodes=[query_node])
    except Exception as e:
        logging.error(f"問候語嵌入計算失敗：{e}")
        return "400: query embedding calculation failed"
    query_embedding = query_embedding_response[0].embedding
    query_embedding_norm = normalize(query_embedding)

    # 使用正規化後的平均向量進行查詢相似度計算
    if hello_document_embedding_norm is not None:
        similarity = cosine_similarity(query_embedding_norm, hello_document_embedding_norm)
        logging.debug(f"問候語與整個文件相關性 (餘弦相似度)：{similarity[0]:.4f}")
    
    # 修正條件判斷
    if hello_normalized_embeddings is not None and len(hello_normalized_embeddings) > 0:
        for i, emb_norm in enumerate(hello_normalized_embeddings):
            sim = cosine_similarity(query_embedding_norm, emb_norm)
            if sim[0] >= 0.9:
                logging.info(f"檢測到問候語（相似度：{sim[0]:.4f}）")
                return True
        logging.debug("未檢測到問候語")
        return False


if __name__ == "__main__":
    # document_activate()
    # document_embedding()
    logging.info("啟動嵌入測試模式")
    while True:
        query = input("請輸入查詢問題：")
        logging.info(f"收到查詢：{query}")
        output_query = queryprocess(query)
        if len(output_query) > 0:
            context_text = "\n".join([f"{context}," for i, context in enumerate(output_query)])
            context_text = context_text.rstrip(",\n")
            logging.info(f"找到 {len(output_query)} 個相關片段")
        else:
            context_text = "找不到與您問題相關的資訊。"
            logging.warning("未找到相關信息")
        print(context_text)
