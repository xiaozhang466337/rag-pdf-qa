# RAG-PDF-QA
轻量化RAG PDF文档问答工具，手动实现RAG核心链路，不依赖LangChain、Chroma、FAISS等重型组件，规避Windows编译依赖问题。

## 功能
- 读取本地PDF文件，自动提取文本
- 文本分片，设置重叠窗口，防止上下文断裂
- 调用通义DashScope Embedding接口向量化
- 手写余弦相似度算法，召回相关性最高的文档片段
- 基于文档内容回答问题，抑制大模型幻觉

## 技术栈
Python
DashScope（通义千问Embedding + Qwen大模型）
PyPDF（PDF文本解析）

## 启动方式
```bash
# 安装依赖
pip install dashscope pypdf

# 运行
python rag_app.py

