import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# -------------------- SETUP --------------------
load_dotenv()

st.set_page_config(page_title="HR Chatbot", layout="wide")
st.title("🤖 HR Policy Chatbot")

# -------------------- LOAD VECTOR DB --------------------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

db = load_db()
retriever = db.as_retriever()

# -------------------- LLM --------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# -------------------- PROMPT --------------------
template = """
Answer the question based on the context:

{context}

Question: {input}
"""

prompt = PromptTemplate.from_template(template)

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# -------------------- CHAT MEMORY --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- DISPLAY OLD MESSAGES --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- USER INPUT --------------------
user_input = st.chat_input("Ask about HR policies...")

if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": user_input})
            answer = response["answer"]

            st.markdown(answer)

    # save response
    st.session_state.messages.append({"role": "assistant", "content": answer})