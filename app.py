from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import os
import json

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")


# Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,environment=PINECONE_API_ENV)

index_name = 'datascience-pilot'

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()

docsearch = Pinecone.from_existing_index(index_name,embeddings)

prompt_template="""
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="F:\My Github Repo\DataSciencePilot\model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                model_type="llama",
                config={'max_new_tokens':512,
                        'temperature':0.8})

qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    chain_type_kwargs=chain_type_kwargs,
    return_source_documents=True)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get',methods=['POST','GET'])
def get_bot_response():
    userText = request.form['msg']
    result=qa({"query": userText})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == "__main__":
    app.run(debug=True, port=5120, host='0.0.0.0')
