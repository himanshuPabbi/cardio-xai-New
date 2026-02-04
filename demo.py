import os
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv


# LangChain & Groq Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.manager import get_openai_callback

# --- 1. INITIALIZE ENVIRONMENT ---
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "heart_2020_cleaned.csv")
LOG_FILE = os.getenv("LOG_FILE", "journal_research_metrics.csv")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_heart_index_full")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="CardioXAI Research", layout="wide")

# --- 2. VECTOR STORE MANAGEMENT ---
@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_DIR):
        return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    
    if not os.path.exists(DATA_PATH):
        st.error("Dataset missing.")
        return None

    st.sidebar.warning("⏳ First Run: Indexing records...")
    df = pd.read_csv(DATA_PATH)
    text_data = df.astype(str).agg(' '.join, axis=1).tolist()
    
    batch_size = 2000 
    vectorstore = None
    progress_bar = st.progress(0)
    
    for i in range(0, len(text_data), batch_size):
        batch = text_data[i : i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_texts(batch, embeddings)
        else:
            vectorstore.add_texts(batch)
        progress_bar.progress(min((i + batch_size) / len(text_data), 1.0))

    vectorstore.save_local(FAISS_INDEX_DIR)
    return vectorstore

# --- 3. LOGGING SYSTEM ---
def log_metrics(query, response, duration, cb, steps):
    code_list = [str(getattr(a, 'tool_input', '')) for a, _ in steps] if steps else []
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": query,
        "llm_response": str(response).replace('\n', ' '), 
        "latency_sec": round(duration, 4),
        "total_tokens": cb.total_tokens,
        "tps": round(cb.completion_tokens / duration, 2) if duration > 0 else 0,
        "generated_code": " | ".join(code_list)
    }
    pd.DataFrame([new_entry]).to_csv(LOG_FILE, mode='a', index=False, header=not os.path.isfile(LOG_FILE), encoding='utf-8')

# --- 4. MAIN APP ---
def main():
    st.title("🔬 CardioXAI: Full-Scale Research Agent")

    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY.")
        return

    if not os.path.exists(DATA_PATH):
        st.error(f"File {DATA_PATH} not found.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # Pre-process binary columns
    binary_cols = [col for col in df.columns if df[col].dtype == 'object' and set(df[col].unique()).issubset({'Yes', 'No'})]
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    get_vector_store()
    
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME, temperature=0.0)

    col_list = ", ".join(df.columns.tolist())
    
    # IMPROVED PREFIX: Strict formatting to prevent "Double Output" hallucination
    custom_prefix = f"""
    You are a Medical Data Scientist working with a pandas dataframe named `df`.
    Available Columns: {col_list}
    Note: Binary columns (Yes/No) are pre-converted to integers (1/0).
    
    CRITICAL FORMATTING RULES:
    1. You MUST use this exact sequence:
       Thought: <reasoning>
       Action: python_repl_ast
       Action Input: <python code>
    2. STOP after the Action Input. Wait for an 'Observation'.
    3. NEVER provide a 'Final Answer' and an 'Action' in the same block.
    4. Provide the 'Final Answer:' ONLY after you have analyzed the result of your code.
    """

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        return_intermediate_steps=True,
        prefix=custom_prefix,
        max_iterations=20,
        max_execution_time=60,
        early_stopping_method="force",
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

    st.markdown("### 📄 Research Batch Processing")
    bulk_input = st.text_area("Enter queries (one per line):", height=150)
    
    if st.button("🚀 Execute Analysis"):
        queries = [q.strip() for q in bulk_input.split("\n") if q.strip()]
        for idx, prompt in enumerate(queries):
            st.markdown(f"**Query {idx+1}:** {prompt}")
            with st.chat_message("assistant"):
                with get_openai_callback() as cb:
                    start_time = time.perf_counter()
                    
                    # RETRY WRAPPER: Specifically handles the "Action + Final Answer" hallucination
                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            result = agent.invoke({"input": prompt})
                            output = result["output"]
                            st.markdown(output)
                            
                            duration = time.perf_counter() - start_time
                            log_metrics(prompt, output, duration, cb, result.get("intermediate_steps", []))
                            break # Success!
                        
                        except Exception as e:
                            error_msg = str(e)
                            if "Parsing LLM output produced both a final answer" in error_msg and attempt < max_retries - 1:
                                time.sleep(0.2)
                                continue # Try again
                            else:
                                st.error(f"Analysis failed: {e}")
                                break
            st.divider()

    if os.path.exists(LOG_FILE):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Metrics")
        log_df = pd.read_csv(LOG_FILE)
        st.sidebar.write(f"Logged: {len(log_df)}")
        with open(LOG_FILE, "rb") as f:
            st.sidebar.download_button("📥 Download Results", f, file_name="research_results.csv")

if __name__ == "__main__":
    main()