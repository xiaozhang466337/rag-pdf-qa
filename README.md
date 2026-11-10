# RAG-PDF-QA
基于LangChain + Chroma向量库 + 通义千问的PDF文档问答助手

## 功能
- 上传PDF文件
- 文档自动切分、向量化存储
- 基于文档内容问答，返回原文片段

## 技术栈
Python、LangChain、ChromaDB、Streamlit、DashScope Embedding

## 启动方式
```bash
# 安装依赖
py -m pip install langchain langchain-community chromadb streamlit dashscope

# 运行
py -m streamlit run rag_app.py
