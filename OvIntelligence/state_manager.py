# state_manager.py
import streamlit as st
import getpass
from typing import Any, Dict
from rate_limiter import RateLimiter
from query_cache import QueryCache

DEFAULT_STATE: Dict[str, Any] = {
    "ollama_ready": False,
    "models_installed": False,
    "query_history": [],
    "rate_limiter": None,
    "query_cache": None,
    "config": {
        "file_selection_model": "qwen2.5-coder:3b",
        "query_processing_model": "qwen2.5-coder:7b",
        "rate_limit": 5,  # seconds between queries
        "paper_checker_mode": False
    },
    "current_user": getpass.getuser(),
    "available_packages": [
        "cellrank_notebooks",
        "scanpy_tutorials",
        "scvi_tutorials",
        "spateo_tutorials",
        "squidpy_notebooks",
        "ov_tut"
    ],
    "selected_package": "cellrank_notebooks",
    "public_url": None,
    "online_search_history": [],
    "paper_content": "",
    "paper_stage1_results": [],
    "llm_temp": 0.5,
    "llm_top_p": 0.5,
}

def initialize_session_state() -> None:
    """
    Ensure that all required keys exist in st.session_state with default values.
    This function is safe to call multiple times.
    """
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Initialize rate limiter and query cache if they are still None
    if st.session_state.get("rate_limiter") is None:
        st.session_state["rate_limiter"] = RateLimiter(st.session_state["config"]["rate_limit"])
    if st.session_state.get("query_cache") is None:
        st.session_state["query_cache"] = QueryCache()

def reset_session_state() -> None:
    """
    Reset the session state to its default values.
    Use this when you need to clear any accumulated history or settings.
    """
    st.session_state.clear()
    initialize_session_state()