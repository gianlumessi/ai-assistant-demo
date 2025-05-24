import streamlit as st
from langchain_community.document_loaders import TextLoader, PDFMinerLoader, Docx2txtLoader
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile
import os
from pathlib import Path

# Streamlit UI
st.title("Company AI Assistant 🤖")

openai_key = st.text_input("🔑 Enter your OpenAI API key to start:", type="password")
if not openai_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_key

uploaded_file = st.file_uploader("📄 Upload a document (TXT, PDF, or DOCX)", type=["txt", "pdf", "docx"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    ext = uploaded_file.name.split(".")[-1].lower()
    loader = {
        "txt": TextLoader,
        "pdf": PDFMinerLoader,
        "docx": Docx2txtLoader
    }.get(ext)

    if not loader:
        st.error("Unsupported file type.")
        st.stop()

    documents = loader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectordb = Chroma.from_documents(chunks, OpenAIEmbeddings())

    user_query = st.text_input("❓ Ask a question based on your document:")
    if user_query:
        docs = vectordb.similarity_search(user_query, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)

        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the context below to answer the question. "
                       "If the answer isn't clear, say 'I don't know.'"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ]).format(context=context, question=user_query)

        response = llm.invoke(prompt)
        st.markdown("### ✅ Answer:")
        st.write(response.content)
