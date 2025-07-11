from flask import Flask, render_template, request, Response
from flask_cors import CORS  # Import CORS
from dotenv import load_dotenv
import os
import time
# import torch
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings
# from langchain_core.runnables import RunnablePassthrough
# import re
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import ChatOllama

# import re
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # 初始化嵌入模型
# local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# # 初始化文本分割器
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

# # 指定包含PDF文件的文件夹路径
# pdf_folder = "./pdfs"  # 替换为你的文件夹路径

# # 存储所有文档的列表
# all_docs = []

# # 遍历文件夹中的所有文件
# for filename in os.listdir(pdf_folder):
#     if filename.lower().endswith('.pdf'):
#         file_path = os.path.join(pdf_folder, filename)
#         print(f"Processing: {file_path}")
        
#         # 加载PDF
#         loader = PDFPlumberLoader(file_path)
#         docs = loader.load()
        
#         # 分割文本
#         splits = text_splitter.split_documents(docs)
#         all_docs.extend(splits)

# # 创建向量存储
# if all_docs:
#     vectorstore = Chroma.from_documents(documents=all_docs, embedding=local_embeddings)
#     print("Vector store created successfully with all PDF documents.")
# else:
#     print("No PDF documents found in the specified folder.")

app = Flask(__name__,template_folder='./')
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()







# RAG_TEMPLATE = """
# 你的名字叫王志晓。
# 根据以下信息回答用户问题，如果用户问题与以下信息无关，则根据你原有的知识回答：

# <context>
# {context}
# </context>

# 用户问题：

# {question}"""

# rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# retriever = vectorstore.as_retriever()


# model = ChatOllama(
#     model="deepseek-r1:32b",
# )


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# qa_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | rag_prompt
#     | model
#     | StrOutputParser()
# )



# def qa_chain_func_stream(query):
#     # thinking = False  # 是否在 <think>...</think> 代码块中

#     # for response_chunk in qa_chain.stream(query):
#     #     # 处理每个流返回的 chunk
#     #     if "<think>" in response_chunk:
#     #         thinking = True  # 开始忽略内容
#     #         continue  # 跳过当前 chunk，不输出

#     #     if "</think>" in response_chunk:
#     #         thinking = False  # 结束忽略内容
#     #         continue  # 跳过当前 chunk，不输出

#     #     if not thinking:
#     #         # print(response_chunk)  # 仅输出非 <think> 块的内容
#     #         yield response_chunk  # 逐块返回数据
#     response =  qa_chain.invoke(query)
#     cleaned_content = re.sub(r"<think>.*?</think>\n?", "", response, flags=re.DOTALL) 
#     return cleaned_content



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/documents')
def index():
    return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.get_json().get('message', '')
    
#     res = qa_chain_func_stream(user_message)
#     return Response(res, content_type='text/event-stream')


# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.get_json().get('message', '')
    
#     res = qa_chain_func_stream(user_message)
#     return Response(res, content_type='text/event-stream')

if __name__ == '__main__':
   # for local loopback only, on dev machine
   # app.run(host='127.0.0.1', port=5000, debug=True)
   # if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
   # listening on all IPs, on server
   app.run(host='0.0.0.0', port=8080, use_reloader=False, debug=True)

