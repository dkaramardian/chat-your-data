import pickle
import os
from query_data import get_chain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS, Qdrant, Chroma



#os.environ["OPENAI_API_KEY"] = put sk key here

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.load_local("vectorstore_regs", embeddings)
    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Chat with your docs!")
    while True:
        print("Human:")
        question = input()
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print("AI:")
        print(result["answer"])
