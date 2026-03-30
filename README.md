# 🤖 RAG Chatbot on Indian Labour & Mines Policy

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that enables users to query and understand complex Indian labour and mining policy documents using natural language.

---

## 🚀 Project Overview

Government policy documents such as labour laws and mining regulations are often lengthy, complex, and difficult to navigate.

This project solves that problem by building an intelligent chatbot that:

* Reads policy documents (PDFs)
* Understands their content using embeddings
* Retrieves relevant sections
* Generates accurate, context-aware answers using an LLM

---

## 🧠 Key Features

* 📄 PDF-based Knowledge Base (Indian Labour & Mines Policies & others)
* 🔍 Semantic Search using FAISS
* 🧩 Intelligent chunking for better retrieval
* 🧠 LLM-powered answer generation (Gemini)
* 💬 ChatGPT-style UI using Streamlit
* ⚡ Fast and scalable (local embeddings – no API limits)

---

## 🏗️ Architecture

User Query
↓
Embedding (HuggingFace MiniLM)
↓
FAISS Vector Search
↓
Relevant Chunks Retrieved
↓
LLM (Gemini)
↓
Final Answer

---

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** Streamlit
* **LLM:** Google Gemini
* **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
* **Vector Database:** FAISS
* **Orchestration:** LangChain

---

## 📂 Project Structure

RAG-chatbot/
│
├── hr/
│   ├── hrpc-FAISS.py        # Build FAISS index
│   ├── hrpc-query.py        # Query pipeline
│   ├── policies/            # PDF documents
│
├── app.py                   # Streamlit chatbot UI
├── faiss_index/             # Stored vector database
├── .env                     # API keys
└── requirements.txt

---

## ⚙️ How It Works

### 1. Document Ingestion

* PDF files are loaded using PyPDFLoader
* Split into smaller chunks using RecursiveCharacterTextSplitter

### 2. Embedding

* Each text chunk is converted into a vector using:
  all-MiniLM-L6-v2 (HuggingFace)

### 3. Vector Storage

* All embeddings are stored in FAISS for fast similarity search

### 4. Retrieval

* User query is converted into embedding
* FAISS returns the most relevant chunks

### 5. Generation

* Retrieved context is passed to Gemini LLM
* LLM generates a final contextual answer

---

## ▶️ Installation & Setup

### 1. Clone Repository

git clone <your-repo-link>
cd RAG-chatbot

### 2. Create Virtual Environment

python -m venv llm_env
llm_env\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Add API Key

Create a `.env` file:

GOOGLE_API_KEY=your_api_key_here

---

## 📊 Build Vector Database

python FAISS.py

---

## 💬 Run Chatbot

streamlit run app.py

---

## 🧪 Example Questions

* What are leave policies under labour law?
* What are safety regulations in mines?
* What are employee rights under Indian labour law?

---

## 📌 Key Learnings

* Built a complete RAG pipeline from scratch
* Implemented vector similarity search using FAISS
* Integrated LLM with retrieval for accurate responses
* Designed a real-world AI system with UI

---

## 🔥 Future Improvements

* 📂 Upload PDF directly from UI
* 📄 Show source citations (page numbers)
* 🌐 Multi-language support
* 🧠 Fully offline LLM (no API dependency)
* 🔎 Hybrid search (BM25 + vector search)


---

## 🤝 Contributing

Feel free to fork this repository and improve the project.

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Nishant Bilagi
Mechanical Engineer → AI/ML Engineer 🚀
