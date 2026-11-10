import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import DashScope
from langchain.chains import RetrievalQA
import os

# ========= 在这里填入你的通义千问API KEY =========
os.environ["DASHSCOPE_API_KEY"] = "你的key放这里"

# 页面配置
st.set_page_config(page_title="PDF知识库问答RAG", layout="wide")
st.title("📖 基于RAG的PDF文档问答助手")

# 初始化embedding模型
embedding = DashScopeEmbeddings(model="text-embedding-v2")
# 大模型
llm = DashScope(model_name="qwen-turbo")

# 文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    length_function=len
)

# 上传PDF
uploaded_file = st.file_uploader("上传PDF文件", type="pdf")

if uploaded_file is not None:
    # 保存临时pdf
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF上传成功，正在解析文档...")

    # 加载文档
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    # 切分
    split_docs = text_splitter.split_documents(docs)
    st.info(f"文档切分为 {len(split_docs)} 个文本块")

    # 构建向量库
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # 提问框
    question = st.text_input("请向文档提问：")
    if question:
        with st.spinner("正在检索文档并生成答案..."):
            result = qa_chain({"query": question})
            st.write("### ✅ 回答：")
            st.write(result["result"])
            with st.expander("查看检索到的原文片段"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)
else:
    st.info("请上传PDF文件开始使用")
