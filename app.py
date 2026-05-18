"""
app.py
------
Streamlit Frontend Client for the Insurance RAG System.
Features:
- Conversational Chat with Sliding Window Memory.
- Advanced RAG Tuning (Top-K Sliders).
- Complete Admin Dashboard (Ingestion, Deletion, and Collection Management).
"""

import os
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration & State Initialization
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Insurance Policy RAG", page_icon="🏥", layout="wide")

# API endpoint from environment or default local
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
ADMIN_PWD = os.environ.get("ADMIN_PASSWORD", "admin123")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Admin Login State
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False


# ---------------------------------------------------------------------------
# API Communication Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=180)  # Cache briefly to prevent redundant API calls
def fetch_collections():
    """Fetches the list of available insurance policy collections from the API."""
    try:
        res = requests.get(f"{API_BASE_URL}/api/v1/admin/collections", timeout=5)
        res.raise_for_status()
        collections = res.json().get("collections", [])
        return sorted(collections)
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
        return []


available_policies = fetch_collections()

# ---------------------------------------------------------------------------
# UI Sidebar - Navigation & Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🏥 RAG Configuration")
    mode = st.radio("Analysis Mode", ["Query Single Policy", "Compare Two Policies"])

    st.divider()

    st.markdown("### Selection")
    if mode == "Query Single Policy":
        selected_policy = st.selectbox("Select Policy", available_policies)
    else:
        # Default to first two if available
        p1_idx = 0 if len(available_policies) > 0 else None
        p2_idx = 1 if len(available_policies) > 1 else 0
        policy_a = st.selectbox(
            "Policy A", available_policies, index=p1_idx, key="pol_a"
        )
        policy_b = st.selectbox(
            "Policy B", available_policies, index=p2_idx, key="pol_b"
        )

    st.divider()

    with st.expander("⚙️ Advanced RAG Tuning"):
        st.info(
            "Adjust these to balance between broader context and strict legal accuracy."
        )
        ret_k = st.slider(
            "Retrieval K (Broad Match Pool)",
            5,
            50,
            15,
            help="Number of chunks pulled from the database initially.",
        )
        rerank_k = st.slider(
            "Rerank K (LLM Context Window)",
            1,
            10,
            5,
            help="Final number of top chunks sent to the LLM for adjudication.",
        )

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main UI Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_admin = st.tabs(["💬 Policy Chatbot", "🛠️ Admin Dashboard"])

# ===========================================================================
# TAB 1: CONVERSATIONAL CHAT
# ===========================================================================
with tab_chat:
    # Display existing chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if query := st.chat_input("What is the maternity waiting period for this policy?"):
        # 1. Add user message to UI and State
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 2. Call backend API
        with st.chat_message("assistant"):
            with st.spinner("Analyzing legal clauses and limits..."):
                try:
                    # IMPLEMENT SLIDING WINDOW: Only send the last 6 messages as history
                    # This ensures the LLM has immediate context without hitting token limits.
                    history_payload = (
                        st.session_state.messages[-7:-1]
                        if st.session_state.messages
                        else []
                    )

                    if mode == "Query Single Policy":
                        if not selected_policy:
                            st.error("Please select a policy from the sidebar first.")
                            st.stop()

                        payload = {
                            "query": query,
                            "collection_name": selected_policy,
                            "history": history_payload,
                            "retrieve_top_k": ret_k,
                            "rerank_top_k": rerank_k,
                        }
                        res = requests.post(
                            f"{API_BASE_URL}/api/v1/query/single", json=payload
                        )
                    else:
                        if policy_a == policy_b:
                            st.warning("Cannot compare a policy against itself.")
                            st.stop()

                        payload = {
                            "query": query,
                            "collection_a": policy_a,
                            "collection_b": policy_b,
                            "history": history_payload,
                            "retrieve_top_k": ret_k,
                            "rerank_top_k": rerank_k,
                        }
                        res = requests.post(
                            f"{API_BASE_URL}/api/v1/query/compare", json=payload
                        )

                    res.raise_for_status()
                    answer = res.json()["data"]["markdown_report"]

                    # 3. Display Assistant Response and Save to State
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    st.error(f"Engine Error: {e}")

# ===========================================================================
# TAB 2: ADMIN DASHBOARD (Management & Ingestion)
# ===========================================================================
with tab_admin:
    if not st.session_state.admin_logged_in:
        st.markdown("### Admin Login Required")
        pwd = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PWD:
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    else:
        st.markdown("### 🛠️ Database Management")
        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

        st.divider()

        # Section A: Existing Collections Manager
        st.markdown("#### 📁 Current Document Collections")
        if not available_policies:
            st.info("The database is currently empty.")
        else:
            for c_name in available_policies:
                col_n, col_d = st.columns([4, 1])
                with col_n:
                    st.write(f"**{c_name}**")
                with col_d:
                    if st.button("Delete", key=f"del_{c_name}", type="primary"):
                        try:
                            res = requests.delete(
                                f"{API_BASE_URL}/api/v1/admin/collections/{c_name}"
                            )
                            res.raise_for_status()
                            st.success(f"Deleted {c_name}")
                            fetch_collections.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

        st.divider()

        # Section B: Ingestion Pipeline
        st.markdown("#### 📥 Add New Document")
        doc_key = (
            st.text_input(
                "Collection Key",
                placeholder="insurance_care_supreme_2024",
                help="Must be lowercase and alphanumeric. The system will automatically prepend 'insurance_' if you omit it.",
            )
            .strip()
            .lower()
        )

        ingest_method = st.radio(
            "Source Type", ["File Upload", "Web URL"], horizontal=True
        )

        # Input placements
        if ingest_method == "File Upload":
            st.file_uploader("Upload PDF Policy", type=["pdf"], key="file_uploader")
        else:
            st.text_input("Direct PDF URL", key="url_input")

        # --- NEW: Advanced Chunking Settings UI ---
        with st.expander("⚙️ Advanced Chunking Settings"):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                # Using pre-filled values based on typical defaults
                input_chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=500,
                    max_value=5000,
                    value=2200,
                    step=100,
                    help="Target characters per chunk.",
                )
            with col_c2:
                input_chunk_overlap = st.number_input(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=1000,
                    value=400,
                    step=50,
                    help="Overlapping characters to maintain context between chunks.",
                )

        if st.button("Process & Index Document", type="primary", disabled=not doc_key):
            # Enforce naming convention
            target_col = (
                doc_key if doc_key.startswith("insurance_") else f"insurance_{doc_key}"
            )

            with st.spinner("Extracting and indexing document..."):
                try:
                    if ingest_method == "File Upload":
                        uploaded_file = st.session_state.get("file_uploader")
                        if not uploaded_file:
                            st.error("Please upload a file.")
                            st.stop()

                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        }
                        # Add chunking params to the form data payload
                        data = {
                            "collection_name": target_col,
                            "chunk_size": input_chunk_size,
                            "chunk_overlap": input_chunk_overlap,
                        }
                        res = requests.post(
                            f"{API_BASE_URL}/api/v1/admin/ingest/file",
                            files=files,
                            data=data,
                        )
                    else:
                        url_val = st.session_state.get("url_input")
                        if not url_val:
                            st.error("Please enter a URL.")
                            st.stop()

                        # Add chunking params to the JSON payload
                        payload = {
                            "url": url_val,
                            "collection_name": target_col,
                            "chunk_size": input_chunk_size,
                            "chunk_overlap": input_chunk_overlap,
                        }
                        res = requests.post(
                            f"{API_BASE_URL}/api/v1/admin/ingest/url", json=payload
                        )

                    res.raise_for_status()
                    st.success("Indexing Complete!")
                    fetch_collections.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
