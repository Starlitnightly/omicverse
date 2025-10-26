# state_manager.py
import streamlit as st
import getpass
from typing import Any, Dict

from config_manager import ConfigManager
from rate_limiter import RateLimiter
from query_cache import QueryCache

DEFAULT_STATE: Dict[str, Any] = {
    "ollama_ready": False,
    "models_installed": False,
    "query_history": [],
    "rate_limiter": None,
    "query_cache": None,
    "config": {},
    "current_user": getpass.getuser(),
    "available_packages": [],
    "selected_package": "",
    "public_url": None,
    "online_search_history": [],
    "paper_content": "",
    "paper_stage1_results": [],
    "llm_temp": 0.5,
    "llm_top_p": 0.5,
    "available_skills": {},
    "skill_usage_log": [],
    "last_skill_match": None,
}

def initialize_session_state() -> None:
    """
    Ensure that all required keys exist in st.session_state with default values.
    This function is safe to call multiple times.
    """
    config = ConfigManager.load_config()
    available_packages = ConfigManager.get_package_names(config)

    dynamic_defaults = dict(DEFAULT_STATE)
    dynamic_defaults["config"] = config
    if available_packages:
        dynamic_defaults["available_packages"] = available_packages
        default_package = available_packages[0]
    else:
        dynamic_defaults["available_packages"] = []
        default_package = ""
    dynamic_defaults["selected_package"] = config.get(
        "selected_package",
        default_package,
    ) or default_package

    for key, value in dynamic_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Initialize rate limiter and query cache if they are still None
    if st.session_state.get("rate_limiter") is None:
        st.session_state["rate_limiter"] = RateLimiter(st.session_state["config"].get("rate_limit", 5))
    if st.session_state.get("query_cache") is None:
        st.session_state["query_cache"] = QueryCache()

def reset_session_state() -> None:
    """
    Reset the session state to its default values.
    Use this when you need to clear any accumulated history or settings.
    """
    st.session_state.clear()
    initialize_session_state()