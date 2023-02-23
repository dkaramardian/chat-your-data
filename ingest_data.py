from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS, Qdrant, Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
import pandas as pd
import pickle
import os

#os.environ["OPENAI_API_KEY"] = put sk key here

#load data and metadata
regdata = pd.read_csv('RegData_CFR_Transportation.csv')
data = list(regdata['documentText'])
sources = list(regdata['documentID'])

#split up docs into chunks
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = ''.join(data)
metadatas = []
texts = text_splitter.split_text(docs)

#create embeddings and vectorstore
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=[{"source": i} \
                                                                for i in range(len(texts))])
                                                                
#save vectorstore to local path
vectorstore.save_local("vectorstore_regs")                                                                
                                                                
