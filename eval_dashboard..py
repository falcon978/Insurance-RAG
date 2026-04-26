import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

st.title("📊 RAG Evaluation Dashboard")
st.markdown("Local analysis of retrieval and generation performance.")

# Load data safely
@st.cache_data
def load_data():
    try:
        with open("eval_results.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

results = load_data()

if not results:
    st.error("No `eval_results.json` found. Please run the DeepEval test suite first.")
    st.stop()

test_cases = results.get("testCases", [])

# Top-level metrics
passing = sum(1 for tc in test_cases if tc["success"])
failing = len(test_cases) - passing

st.write("### Pipeline Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Golden Queries", len(test_cases))
col2.metric("Passed All Thresholds", passing)
col3.metric("Failed", failing)

st.divider()

# Interactive filtering sidebar
st.sidebar.header("Filters")
status_filter = st.sidebar.radio("Filter by Status:", ["All", "Pass", "Fail"])

# Display detailed results
st.write("### Granular Query Analysis")

for i, tc in enumerate(test_cases):
    is_pass = tc["success"]
    
    # Apply filter
    if status_filter == "Pass" and not is_pass: continue
    if status_filter == "Fail" and is_pass: continue
        
    status_icon = "✅" if is_pass else "❌"
    
    with st.expander(f"{status_icon} Query {i+1}: {tc['input'][:80]}..."):
        st.markdown("**User Query:**")
        st.info(tc["input"])
        
        st.markdown("**RAG Actual Output:**")
        st.write(tc["actualOutput"])
        
        st.markdown("**Metrics Breakdown:**")
        
        # Build a table for the metrics
        metrics_data = []
        for metric in tc.get("metrics", []):
            metrics_data.append({
                "Metric Name": metric["name"],
                "Score": round(metric["score"], 2),
                "Threshold": metric["threshold"],
                "Status": "✅ Pass" if metric["success"] else "❌ Fail",
                "Judge's Reasoning": metric["reason"]
            })
            
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)