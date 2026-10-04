# --- SQLite fix for Streamlit Cloud (keep this at the very top) ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import streamlit as st
from learn_mate_crew import run_research

st.set_page_config(page_title="Learn Mate AI", page_icon="📚")
st.title("📚 Learn Mate AI")
st.caption("Enter a topic and get a researched report.")

# Read the Groq API key from Streamlit secrets only
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error(
        "GROQ_API_KEY not found. In Streamlit Cloud, open your app's "
        "Settings → Secrets and add:  GROQ_API_KEY = \"your_key\""
    )
    st.stop()

topic = st.text_input("Research topic", placeholder="e.g. How do vaccines work?")

if st.button("Generate Report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Searching and writing your report... (can take 30-90 seconds)"):
            try:
                st.session_state["report"] = run_research(topic.strip(), api_key)
            except Exception as e:
                st.error(f"Something went wrong: {e}")

if "report" in st.session_state:
    st.markdown(st.session_state["report"])
    st.download_button(
        "Download report (.md)",
        st.session_state["report"],
        file_name="learn_mate_report.md",
        mime="text/markdown",
    )

    
