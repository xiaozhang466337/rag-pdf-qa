# 把你全部RAG代码粘贴在这里
!pip install dashscope pypdf openai numpy
import dashscope
from dashscope import TextEmbedding, Generation
from pypdf import PdfReader

dashscope.api_key = "填入你的DashScope Key"
dashscope.base_http_api_url = "填入你的借楼兼容地址"
import dashscope
from dashscope import TextEmbedding
from pypdf import PdfReader
import numpy as np
from openai import OpenAI
# ========== 配置区，改成你自己的密钥 ==========
API_KEY = "填入你的DashScope Key"
BASE_URL = "填入你的借楼兼容地址"
PDF_NAME = "加油.pdf"
# ===========================================
# 初始化大模型客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 文本切分函数
def split_text(text, chunk_size=500, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# 获取embedding
def get_embedding(text):
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v4,
        input=[text]
    )
    if resp.status_code != 200:
        print("❌ Embedding调用失败")
        return None
    return resp.output['embeddings'][0]['embedding']
    # 余弦相似度
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 读取PDF
reader = PdfReader(PDF_NAME)
full_text = ""
for page in reader.pages:
    txt = page.extract_text()
    if txt:
        full_text += txt

# 分块 + 向量化
chunks = split_text(full_text)
chunk_embeds = []
print("正在生成文本向量，请稍等...")
for c in chunks:
    emb = get_embedding(c)
    if emb is not None:
        chunk_embeds.append((c, emb))
print(f"✅ 文档处理完成，一共 {len(chunk_embeds)} 个文本块")
# 问答循环
print("\n===== RAG问答系统启动，输入 exit 退出 =====")
while True:
    question = input("你的问题：")
    if question.strip() == "exit":
        print("程序退出")
        break
    q_emb = get_embedding(question)
    # 相似度排序，取top3
    score_list = []
    for c, emb in chunk_embeds:
        score_list.append((cos_sim(q_emb, emb), c))
    score_list.sort(reverse=True)
    top3_context = "\n".join([item[1] for item in score_list[:3]])

    prompt = f"基于下面的参考材料回答问题，优先使用材料里面的内容，不要编造信息：\n【参考材料】\n{top3_context}\n【问题】{question}"
    rsp = client.chat.completions.create(
        model="qwen-flash",
        messages=[{"role":"user", "content": prompt}]
    )
    print("回答：", rsp.choices[0].message.content, "\n")














