import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PDFMinerLoader, Docx2txtLoader
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env from parent directory
env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Streamlit UI
st.title("Company AI Assistant 🤖")
uploaded_file = st.file_uploader("Upload a file (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])

if uploaded_file:
    # Save to a temporary file
    with NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load file content
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext == "txt":
        loader = TextLoader(tmp_path)
    elif file_ext == "pdf":
        loader = PDFMinerLoader(tmp_path)
    elif file_ext == "docx":
        loader = Docx2txtLoader(tmp_path)
    else:
        st.error("Unsupported file type.")
        st.stop()

    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Create vector store
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embedding)

    # Question input
    user_query = st.text_input("Ask a question based on your document:")

    if user_query:
        # Retrieve relevant chunks
        docs = vectordb.similarity_search(user_query, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Chat completion
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        template = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Use the context below to answer the question. "
             "If the answer isn't clear, say 'I don't know.'"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        prompt = template.format(context=context, question=user_query)
        response = llm.invoke(prompt)

        st.markdown("### Answer:")
        st.write(response.content)
