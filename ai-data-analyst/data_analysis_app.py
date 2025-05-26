import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os

# --- UI Setup ---
st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("📊 AI Data Analyst")
st.write("Ask questions about your Excel file and get instant insights with AI + charts.")

# --- Sidebar ---
st.sidebar.title("🔧 Settings")

st.sidebar.markdown("""
Upload your Excel file or use the sample data.  
Enter your OpenAI API key to generate answers using GPT-3.5 Turbo.
""")
# User input
openai_key = st.sidebar.text_input("🔐 OpenAI API Key", type="password")

use_sample = st.sidebar.checkbox("📁 Use sample dataset (Walmart_sales.xlsx)?")

uploaded_file = None
if not use_sample:
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

# --- Load Data ---
df = None
if use_sample:
    sample_path = os.path.join(os.path.dirname(__file__), "Walmart_sales.xlsx")
    df = pd.read_excel(sample_path)
elif uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

# --- Show preview ---
if df is not None:
    st.success("✅ Data loaded successfully.")
    st.dataframe(df.head())

    # --- Question Input ---
    question = st.text_input("💬 Ask a question about your data")

    if question and openai_key:
        # --- Create Agent ---
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type="openai-tools",
            allow_dangerous_code=True
        )

        # --- Run GPT Agent ---
        with st.spinner("Thinking..."):
            try:
                response = agent.run(question)
                st.markdown("### 💡 Answer:")
                st.success(response)

                # Show chart if one was generated
                fig = plt.gcf()
                if fig.axes:
                    st.markdown("### 📊 Chart:")
                    for ax in fig.axes:
                        for label in ax.get_xticklabels():
                            label.set_rotation(45)
                            label.set_fontsize(12)
                        for label in ax.get_yticklabels():
                            label.set_fontsize(12)
                        ax.xaxis.label.set_fontsize(14)
                        ax.yaxis.label.set_fontsize(14)
                        if ax.title:
                            ax.title.set_fontsize(16)

                    st.pyplot(fig)
                    plt.clf()
            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif question and not openai_key:
        st.warning("Please enter your OpenAI API key to ask a question.")
