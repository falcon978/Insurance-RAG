"""
app.py
------
Streamlit Interface for the Insurance RAG System.
Features a public Chatbot tab and a secured Admin tab for document ingestion.
"""

import os
import time
import tempfile
import urllib.request
import streamlit as st
import chromadb

# Import your existing pipeline and the NEW dynamically decoupled RAG engine
from pipeline import ExtractionPipeline
from engine import InsuranceRAGEngine

# ---------------------------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Insurance Policy RAG", page_icon="🏥", layout="wide")

CHROMA_DIR = "./chroma_data"
ADMIN_PWD = os.environ.get("ADMIN_PASSWORD", "admin123")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Cache the Chroma client
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)

client = get_chroma_client()

# Cache the new Orchestrator Engine
@st.cache_resource
def get_rag_engine():
    if not GEMINI_API_KEY:
        st.warning("GEMINI_API_KEY environment variable is not set. Generation will fail.")
    return InsuranceRAGEngine(gemini_api_key=GEMINI_API_KEY, chroma_dir=CHROMA_DIR)

# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

st.title("🏥 Insurance Policy Analyzer")

tab_chat, tab_admin = st.tabs(["💬 Chat", "⚙️ Admin Dashboard"])

# ===========================================================================
# TAB 1: PUBLIC CHAT 
# ===========================================================================
with tab_chat:
    st.markdown("### Policy Chatbot")
    
    collections = [c.name for c in client.list_collections()]
    
    if not collections:
        st.warning("No policies indexed yet. Go to the Admin tab to upload one.")
    else:
        rag = get_rag_engine()
        
        # 2. UI Controls
        st.markdown("#### Analysis Mode")
        mode = st.radio("Select Mode", ["Query Single Policy", "Compare Two Policies"], horizontal=True, label_visibility="collapsed")
        
        st.markdown("#### Policy Selection")
        if mode == "Query Single Policy":
            selected_policy = st.selectbox("Select Policy", collections)
        else:
            col1, col2 = st.columns(2)
            policy_a = col1.selectbox("Select First Policy", collections, key="pol_a")
            policy_b = col2.selectbox("Select Second Policy", collections, key="pol_b")
            
        with st.expander("⚙️ Advanced Tuning"):
            ret_k = st.slider("Retrieval Pool (top_k)", 5, 50, 15, help="Number of chunks pulled from ChromaDB + BM25")
            rerank_k = st.slider("LLM Context Chunks", 1, 10, 3, help="Final number of highly relevant chunks sent to the LLM")
                
        st.divider()
        
        # 3. Chat Input & Processing
        query = st.chat_input("Ask a question (e.g., 'What is the maternity waiting period?')")
        
        if query:
            with st.chat_message("user"):
                st.write(query)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing legal clauses and limits..."):
                    try:
                        if mode == "Query Single Policy":
                            # ---> PASS THE SLIDER VALUES HERE <---
                            answer = rag.query_single_policy(
                                query, 
                                collection_name=selected_policy,
                                retrieve_top_k=ret_k,
                                rerank_top_k=rerank_k
                            )
                        else:
                            if policy_a == policy_b:
                                st.error("Cannot compare a policy against itself.")
                                st.stop()
                            # ---> PASS THE SLIDER VALUES HERE <---
                            answer = rag.compare_policies(
                                query, 
                                collection_a=policy_a, 
                                collection_b=policy_b,
                                retrieve_top_k=ret_k,
                                rerank_top_k=rerank_k
                            )
                            
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"An error occurred while generating the response: {e}")

# ===========================================================================
# TAB 2: ADMIN DASHBOARD (Unchanged - Works perfectly)
# ===========================================================================
# [Keep your exact Tab 2 Admin code here. It does not need any changes!]
with tab_admin:
    
    # --- Authentication Check ---
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        st.markdown("### Admin Login Required")
        pwd = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PWD:
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    
    # --- Authenticated Admin Area ---
    else:
        col_title, col_logout = st.columns([4, 1])
        with col_title:
            st.markdown("### 🛠️ Database Management")
        with col_logout:
            if st.button("Logout"):
                st.session_state.admin_logged_in = False
                st.rerun()

        st.divider()

        # ---------------------------------------------------------
        # Section A: Existing Collections Manager
        # ---------------------------------------------------------
        st.markdown("#### 📁 Current Document Collections")
        
        collections = client.list_collections()
        
        if not collections:
            st.info("The database is currently empty. Add a document below.")
        else:
            for c in collections:
                # Streamlit layout tricks for a clean list with delete buttons
                col_name, col_count, col_btn = st.columns([3, 2, 1])
                with col_name:
                    st.write(f"**{c.name}**")
                with col_count:
                    st.write(f"{c.count()} chunks indexed")
                with col_btn:
                    if st.button("Delete", key=f"del_{c.name}", type="primary"):
                        client.delete_collection(c.name)
                        st.success(f"Deleted collection: {c.name}")
                        st.rerun()

        st.divider()

        # ---------------------------------------------------------
        # Section B: Ingestion Pipeline
        # ---------------------------------------------------------
        st.markdown("#### 📥 Add New Document")
        
        # The key used to name the Chroma collection
        doc_key = st.text_input("Collection Key (e.g., 'hdfc_optima_2025')").strip().lower()
        
        # Ensure the key uses valid characters for ChromaDB
        if doc_key and not doc_key.replace("_", "").isalnum():
            st.warning("Please use only alphanumeric characters and underscores for the key.")
            doc_key = ""

        ingest_method = st.radio("Source Type", ["File Upload", "Web URL"], horizontal=True)

        # Container to hold the dynamic input (File uploader or Text input)
        input_container = st.container()
        
        if st.button("Process & Index Document", type="primary", disabled=not doc_key):
            target_collection = f"insurance_{doc_key}"
            
            with st.spinner(f"Extracting and indexing into '{target_collection}'... This may take a minute."):
                tmp_path = None
                
                try:
                    # Handle File Upload
                    if ingest_method == "File Upload":
                        uploaded_file = st.session_state.get("file_uploader")
                        if not uploaded_file:
                            st.error("Please select a file first.")
                            st.stop()
                            
                        # Save the Streamlit uploaded file to a temporary disk location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                    # Handle Web URL Download
                    elif ingest_method == "Web URL":
                        url = st.session_state.get("url_input")
                        if not url:
                            st.error("Please enter a valid URL.")
                            st.stop()
                            
                        # Download the PDF to a temporary disk location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req, timeout=60) as r:
                                tmp.write(r.read())
                            tmp_path = tmp.name

                    # 🚀 Run Your Master Pipeline!
                    if tmp_path:
                        ExtractionPipeline(
                            pdf_path=tmp_path,
                            persist_dir=CHROMA_DIR,
                            collection_name=target_collection,
                            verbose=False # Keep the terminal clean
                        ).run()
                        
                        st.success(f"✅ Successfully indexed document into '{target_collection}'!")
                        time.sleep(2) # Show the success message briefly before refreshing
                        st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during ingestion: {e}")
                
                finally:
                    # Always clean up the temporary file
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        # Place the inputs inside the container defined above (Streamlit layout flow)
        with input_container:
            if ingest_method == "File Upload":
                st.file_uploader("Upload PDF Policy", type=["pdf"], key="file_uploader")
            else:
                st.text_input("Direct PDF URL", key="url_input")