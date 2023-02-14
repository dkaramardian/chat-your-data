from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
#from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS, Qdrant, Chroma
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os

#os.environ["OPENAI_API_KEY"] = put sk key here

# Load Data
loader = UnstructuredFileLoader("Sample_processed_regdata.txt")
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)


# Load Data to vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)


# Save vectorstore
with open("vectorstore3.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
