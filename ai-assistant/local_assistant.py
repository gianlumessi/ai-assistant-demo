import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
import os
from pathlib import Path


# Load .env from parent directory
env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API Key here
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load and split the document
loader = TextLoader("faq.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
texts = text_splitter.split_documents(documents)

# Embed and store
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embedding)

# Set up Streamlit UI
st.title("Simple AI Assistant 🤖")
user_query = st.text_input("Ask a question:")

if user_query:
    # Find similar documents
    docs = vectordb.similarity_search(user_query, k=4)

    # Combine the content
    context = "\n\n".join(doc.page_content for doc in docs)

    # Send to GPT
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful company assistant. Use the provided context to answer the user's question as accurately as possible. "
         "If the answer is unclear from the context, use your best judgment based on the information you have and say you don't know if you are not confident of your answer."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    prompt = template.format(context=context, question=user_query)
    response = llm.invoke(prompt)

    st.write(response.content)
