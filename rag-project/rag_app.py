import os
import dashscope
from dashscope import TextEmbedding
from pypdf import PdfReader

# 读取环境变量里的密钥
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("请先在PowerShell设置DASHSCOPE_API_KEY环境变量！")
    exit()
dashscope.api_key = api_key

# 文本切分函数
def split_text(text, chunk_size=500, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# 计算余弦相似度
def cos_sim(vec1, vec2):
    dot = sum(a*b for a,b in zip(vec1,vec2))
    norm1 = sum(a*a for a in vec1)**0.5
    norm2 = sum(a*a for a in vec2)**0.5
    return dot/(norm1*norm2)

# 向量化
def get_embedding(text):
    rsp = TextEmbedding.call(model=TextEmbedding.Models.text_embedding_v2, input=text)
    return rsp.output['embeddings'][0]['embedding']

# 加载PDF
pdf_path = input("输入PDF完整路径：")
reader = PdfReader(pdf_path)
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

chunks = split_text(full_text)
# 给所有块生成向量
chunk_embeds = []
for c in chunks:
    chunk_embeds.append( (c, get_embedding(c)) )

print("\n✅ PDF加载完成，可以提问，输入exit退出！")
while True:
    q = input("\n你的问题：")
    if q.strip().lower() == "exit":
        break
    q_emb = get_embedding(q)
    # 找最相似3块
    score_list = []
    for txt,emb in chunk_embeds:
        score_list.append( (cos_sim(q_emb, emb), txt) )
    score_list.sort(reverse=True)
    top3 = [x[1] for x in score_list[:3]]
    context = "\n".join(top3)

    prompt = f"""基于下面文档回答，不要编造内容：
【文档】
{context}
【问题】
{q}
"""
    # 调用大模型
    from dashscope import Generation
    resp = Generation.call(
        model=Generation.Models.qwen_turbo,
        messages=[{"role":"user","content":prompt}],
        result_format="message"
    )
    print("回答：", resp.output.choices[0].message.content)
