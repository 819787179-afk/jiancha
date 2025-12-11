import streamlit as st
import json
import pandas as pd
import os
from threading import Lock
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="信息流脚本查重工具", layout="wide")

# -----------------------------
# 多用户密码设置
# -----------------------------
USER_PASSWORDS = {
    "user1": "123456",
    "user2": "abcdef"
}

st.title("信息流脚本查重工具（公网多人版）")

username = st.text_input("用户名")
password_input = st.text_input("密码", type="password")

if username not in USER_PASSWORDS or USER_PASSWORDS[username] != password_input:
    st.warning("用户名或密码错误，请重新输入")
    st.stop()

# -----------------------------
# 老文案库路径和锁
# -----------------------------
OLD_FILE = "old_scripts.json"   # 放在服务器上的老文案库
BACKUP_DIR = "backup_scripts"
lock = Lock()

# 创建备份目录
os.makedirs(BACKUP_DIR, exist_ok=True)

# 加载老文案库
def load_old_scripts():
    texts = []
    if os.path.exists(OLD_FILE):
        with open(OLD_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        texts.append(data['text'])
                    except:
                        continue
    return texts

old_texts = load_old_scripts()

st.subheader(f"老文案库已加载，共 {len(old_texts)} 条文案")

# -----------------------------
# 新文案输入
# -----------------------------
new_text_input = st.text_area(
    "请输入新文案，每条完整文案用空行分隔",
    height=250
)
new_texts = [x.strip() for x in new_text_input.strip().split("\n\n") if x.strip()] if new_text_input else []

# -----------------------------
# 查重逻辑
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

def check_similarity(old_texts, new_texts, threshold=0.85, tfidf_weight=0.7, fuzzy_weight=0.3):
    """
    综合查重逻辑：
    - TF-IDF 向量相似度
    - 句级模糊匹配 (RapidFuzz)
    - 按权重计算综合分数
    """
    if not old_texts or not new_texts:
        return []

    # 1. TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(old_texts + new_texts)
    old_matrix = tfidf_matrix[:len(old_texts)]
    new_matrix = tfidf_matrix[len(old_texts):]

    results = []
    for i, new_vec in enumerate(new_matrix):
        # TF-IDF 相似度
        sims = cosine_similarity(new_vec, old_matrix)
        max_tfidf = sims.max()
        idx_tfidf = sims.argmax()

        # 句级模糊匹配相似度
        fuzzy_scores = [fuzz.token_sort_ratio(new_texts[i], old) / 100 for old in old_texts]
        max_fuzzy = max(fuzzy_scores)
        idx_fuzzy = fuzzy_scores.index(max_fuzzy)

        # 综合得分
        combined_score = tfidf_weight * max_tfidf + fuzzy_weight * max_fuzzy
        # 取 TF-IDF 最相似老文案为参考
        best_idx = idx_tfidf if max_tfidf >= max_fuzzy else idx_fuzzy

        results.append({
            "新文案": new_texts[i],
            "最相似老文案": old_texts[best_idx],
            "相似度": f"{round(combined_score*100, 1)}%",
            "可能重复": "⚠️ 是" if combined_score >= threshold else "否"
        })
    return results

# -----------------------------
# 查重操作
# -----------------------------
threshold = st.slider("重复阈值", 0.0, 1.0, 0.7, 0.01)

if st.button("开始查重"):
    if not old_texts:
        st.warning("老文案库为空，无法查重")
    elif not new_texts:
        st.warning("请输入新文案")
    else:
        results = check_similarity(old_texts, new_texts, threshold)
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        # 导出 Excel
        output_file = f"check_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(output_file, index=False)
        st.success(f"查重完成！结果已保存到 {output_file}")

# -----------------------------
# 累加新文案到老文案库（安全锁定+自动备份）
# -----------------------------
if new_texts and st.button("累加新文案到文献库"):
    with lock:
        # 备份旧文件
        if os.path.exists(OLD_FILE):
            backup_file = os.path.join(
                BACKUP_DIR,
                f"old_scripts_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(OLD_FILE, "r", encoding="utf-8") as f_old, \
                 open(backup_file, "w", encoding="utf-8") as f_backup:
                f_backup.write(f_old.read())
        
        # 追加新文案
        with open(OLD_FILE, "a", encoding="utf-8") as f:
            for text in new_texts:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write("\n")
        st.success(f"文献库已更新并备份，新增 {len(new_texts)} 条文案已累加")
