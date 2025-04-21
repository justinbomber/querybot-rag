import re
import os

def parse_law_articles(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正規表示法匹配 "第 xxx 條"，含 28-1 這類型的條號
    pattern = r'(第 \s*[\d\-]+ 條)'  # 支援中文數字、數字、連字符
    splits = re.split(pattern, content)

    # re.split 會保留 pattern 中的括號匹配結果，第一個元素可能是前導空白文字要忽略
    articles = []
    i = 1
    while i < len(splits) - 1:
        title = splits[i].strip()  # 第 XXX 條
        body = splits[i + 1].strip().replace('\n', '').replace('\r', '')  # 內容合併（移除換行）
        merged = f"{title}{body}"
        articles.append(merged)
        i += 2

    return '\n'.join(articles)

if __name__ == '__main__':
    result = parse_law_articles('./company_law.txt')
    with open('./company_law_parsed.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    # print(result)