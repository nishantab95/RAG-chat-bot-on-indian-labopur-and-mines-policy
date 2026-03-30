from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os


def upload_pdf():
    load_dotenv()
    all_docs = []
    folder_path = 'E:/Git/RAG chatbot/hr/policies'

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"{folder_path}/{file}")
            docs = loader.load()
            all_docs.extend(docs)


    print(f"{len(docs)} Pages loaded")

    # 🔹 Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""])

    split_documents = text_splitter.split_documents(docs)
    print(f"Split into {len(split_documents)} Documents...")

    print(split_documents[0].metadata)

    # 🔹 Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # 🔹 Create FAISS DB
    db = FAISS.from_documents(split_documents, embeddings)

    # 🔹 Save locally
    db.save_local("faiss_index")

    print("✅ FAISS index saved successfully!")

if __name__ == "__main__":
    upload_pdf()