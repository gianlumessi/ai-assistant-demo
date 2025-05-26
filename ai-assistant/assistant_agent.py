import streamlit as st
from langchain_community.document_loaders import TextLoader, PDFMinerLoader, Docx2txtLoader
#from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile
import os

gpt_version = "gpt-3.5-turbo" #or 'gpt-4'

# Sidebar: API key input
st.sidebar.title("🔐 API Key Required")
openai_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

os.environ["OPENAI_API_KEY"] = openai_key

# Sidebar: Info
st.sidebar.markdown("---")
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown(f"""
**What it does:**
- Upload a file (TXT, PDF, or DOCX)
- Ask natural language questions
- Get answers from **{gpt_version}** based on your document

**Is my data safe?**
- ✅ Your file is processed temporarily in memory
- ✅ Your OpenAI key is only used during this session
- ❌ Nothing is stored or shared by this app
""")

if not openai_key:
    st.sidebar.warning("Required to use the assistant.")
    st.stop()

# Main UI
st.title("📄 Company AI Assistant")
uploaded_file = st.file_uploader("Upload a document (TXT, PDF, or DOCX)", type=["txt", "pdf", "docx"])

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
    #vectordb = Chroma.from_documents(chunks, OpenAIEmbeddings())
    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings())

    user_query = st.text_input("❓ Ask a question based on your document:")
    if user_query:
        docs = vectordb.similarity_search(user_query, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)

        llm = ChatOpenAI(model_name=gpt_version, temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the context below to answer the question. "
                       "If the answer isn't clear, say 'I don't know.'"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ]).format(context=context, question=user_query)

        response = llm.invoke(prompt)
        st.markdown("### ✅ Answer:")
        st.write(response.content)
