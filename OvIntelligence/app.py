# --- START OF FILE app.py ---
import streamlit as st
import ast
import os
import subprocess
import time
import requests
import getpass
import psutil
from datetime import datetime, timezone
from pathlib import Path
from collections import OrderedDict

# Use our centralized logger setup
from logger_setup import logger

# Try to import pyngrok for public link support
try:
    from pyngrok import ngrok
    PYNGROK_AVAILABLE = True
except ImportError:
    PYNGROK_AVAILABLE = False

import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

from rag_system import RAGSystem, PackageConfig
from config_manager import ConfigManager
from system_monitor import SystemMonitor
from rate_limiter import RateLimiter
from query_cache import QueryCache
from query_manager import QueryManager

from token_counter import global_token_counter, count_tokens_decorator

# Import session state management
from state_manager import initialize_session_state, reset_session_state

# ---- NEW: Import the external paper checker mode module ----
from paper_checker_mode import paper_checker_interface

###############################################################################
#                       SESSION STATE INITIALIZATION
###############################################################################
# Initialize Streamlit session state with default values.
initialize_session_state()

@st.cache_resource(show_spinner=False, hash_funcs={dict: lambda d: str(sorted(d.items()))})
def get_rag_system(config: dict):
    package_configs = [
        PackageConfig(
            name="cellrank_notebooks",
            converted_jsons_directory="/Users/kq_m3m/PycharmProjects/SCMaster/6O_json_files/cellrank_notebooks",
            annotated_scripts_directory="/Users/kq_m3m/PycharmProjects/SCMaster/annotated_scripts/cellrank_notebooks",
            file_selection_model=config['file_selection_model'],
            query_processing_model=config['query_processing_model']
        ),
        PackageConfig(
            name="scanpy_tutorials",
            converted_jsons_directory="/Users/kq_m3m/PycharmProjects/SCMaster/6O_json_files/scanpy-tutorials",
            annotated_scripts_directory="/Users/kq_m3m/PycharmProjects/SCMaster/annotated_scripts/scanpy-tutorials",
            file_selection_model=config['file_selection_model'],
            query_processing_model=config['query_processing_model']
        ),
        PackageConfig(
            name="scvi_tutorials",
            converted_jsons_directory="/Users/kq_m3m/PycharmProjects/SCMaster/6O_json_files/scvi-tutorials",
            annotated_scripts_directory="/Users/kq_m3m/PycharmProjects/SCMaster/annotated_scripts/scvi-tutorials",
            file_selection_model=config['file_selection_model'],
            query_processing_model=config['query_processing_model']
        ),
        PackageConfig(
            name="spateo_tutorials",
            converted_jsons_directory="/Users/kq_m3m/PycharmProjects/SCMaster/6O_json_files/spateo-tutorials",
            annotated_scripts_directory="/Users/kq_m3m/PycharmProjects/SCMaster/annotated_scripts/spateo-tutorials",
            file_selection_model=config['file_selection_model'],
            query_processing_model=config['query_processing_model']
        ),
        PackageConfig(
            name="squidpy_notebooks",
            converted_jsons_directory="/Users/kq_m3m/PycharmProjects/SCMaster/6O_json_files/squidpy_notebooks",
            annotated_scripts_directory="/Users/kq_m3m/PycharmProjects/SCMaster/annotated_scripts/squidpy_notebooks",
            file_selection_model=config['file_selection_model'],
            query_processing_model=config['query_processing_model']
        ),
        PackageConfig(
            name="ov_tut",
            converted_jsons_directory="/Users/kq_m3m/PycharmProjects/SCMaster/6O_json_files/ov_tut",
            annotated_scripts_directory="/Users/kq_m3m/PycharmProjects/SCMaster/annotated_scripts/ov_tut",
            file_selection_model=config['file_selection_model'],
            query_processing_model=config['query_processing_model']
        ),
    ]
    try:
        return RAGSystem(package_configs)
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}", exc_info=True)
        return None

###############################################################################
#                           PAGE / SIDEBAR FUNCTIONS
###############################################################################
def show_header():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("Agentic OmicVerse 🧬")
    with col2:
        current_time = datetime.now(timezone.utc)
        st.info(f"📅 UTC: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    with col3:
        st.info(f"👤 User: {st.session_state['current_user']}")

def show_system_status():
    stats = SystemMonitor.get_system_stats()
    with st.sidebar:
        st.header("System Status 📊")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Memory (MB)", f"{stats['memory_usage']:.1f}")
            st.metric("CPU %", f"{stats['cpu_percent']:.1f}")
        with col2:
            st.metric("Uptime", SystemMonitor.format_uptime(stats['uptime']))
            st.metric("Memory Usage %", f"{stats['system_memory']['percent']:.1f}")
        st.progress(stats['system_memory']['percent'] / 100)


def check_ollama_server() -> bool:
    """Check if Ollama server is running and responsive with improved diagnostics and retries."""
    max_retries = 3
    timeout = 10  # Increased from 5 seconds

    for attempt in range(max_retries):
        try:
            # Log the attempt for debugging
            logger.info(f"Checking Ollama server (attempt {attempt + 1}/{max_retries})")

            # Try to connect to the API endpoint
            response = requests.get("http://localhost:11434/api/version", timeout=timeout)

            # Log successful response for debugging
            logger.info(f"Ollama server responded with status code: {response.status_code}")

            if response.status_code == 200:
                logger.info("Ollama server is running and responsive")
                return True

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout connecting to Ollama server (attempt {attempt + 1})")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error to Ollama server (attempt {attempt + 1})")
        except requests.RequestException as e:
            logger.warning(f"Request exception when checking Ollama: {str(e)}")

        # Wait before retrying
        if attempt < max_retries - 1:
            time.sleep(2)

    # As a fallback, check if the port is open using socket
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()

        if result == 0:
            logger.info("Port 11434 is open, but API is not responding")
        else:
            logger.info("Port 11434 is not accessible")

    except Exception as e:
        logger.warning(f"Socket check failed: {str(e)}")

    return False

def display_health_status():
    healthy, checks = check_system_health()
    with st.sidebar:
        st.header("System Health ✅" if healthy else "System Health ⚠️")
        for component, status in checks.items():
            if status:
                st.success(f"{component} is running")
            else:
                st.error(f"{component} is not running")

def check_system_health():
    health_checks = {
        'Ollama Server': check_ollama_server(),
    }
    all_healthy = all(health_checks.values())
    return all_healthy, health_checks

def show_query_history():
    with st.sidebar:
        st.header("Query History 📜")
        for idx, item in enumerate(reversed(st.session_state.query_history[-10:])):
            with st.expander(f"Query {len(st.session_state.query_history) - idx}: {item['query'][:30]}..."):
                st.markdown(f"**Package:** {item['package']}")
                st.markdown(f"**Time:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
                st.markdown(f"**User:** {item['user']}")
                st.markdown(f"**Document(s):** {item['file']}")
                st.markdown(f"**Answer:** {item['answer']}")
                st.markdown("---")

def show_online_search_history():
    with st.sidebar:
        st.header("Online Search History 🌐")
        for idx, item in enumerate(reversed(st.session_state['online_search_history'][-10:])):
            with st.expander(f"Online Search {len(st.session_state['online_search_history']) - idx}: {item['query'][:30]}..."):
                st.markdown(f"**Time:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
                st.markdown(f"**User:** {item['user']}")
                st.markdown(f"**Answer:** {item['answer']}")
                st.markdown("---")

def show_public_link_controls():
    st.header("Public Link 🔗")
    if not PYNGROK_AVAILABLE:
        st.error("pyngrok is not installed. Run pip install pyngrok to enable sharing.")
        return

    if st.session_state['public_url'] is None:
        if st.button("🌐 Launch Public Link"):
            try:
                public_url = ngrok.connect(8501, "http")
                st.session_state['public_url'] = public_url.public_url
                st.success(f"Public link is now active! {public_url.public_url}")
            except Exception as e:
                st.error(f"Failed to create a public link: {str(e)}")
    else:
        st.info(f"Public link is active: {st.session_state['public_url']}")
        if st.button("🔒 Stop Public Link"):
            try:
                ngrok.kill()
                st.session_state['public_url'] = None
                st.success("Public link has been stopped.")
            except Exception as e:
                st.error(f"Failed to stop public link: {str(e)}")

###############################################################################
#                           NORMAL (RAG) QUERY INTERFACE
###############################################################################
@count_tokens_decorator
def process_query_with_progress(query, rag_system, selected_package, user="unknown"):
    if not query or not isinstance(query, str):
        raise ValueError("Invalid query: Query must be a non-empty string")
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        logger.info(f"Processing query: {query}")
        logger.info(f"Selected package: {selected_package}")
        status_text.text("Finding relevant documents...")
        progress_bar.progress(25)
        relevant_files = rag_system.find_relevant_files(selected_package, query)
        status_text.text("Generating answer from annotated scripts...")
        progress_bar.progress(50)
        answer = rag_system.answer_query_with_annotated_scripts(
            selected_package, query, relevant_files
        )
        logger.info(f"Answer: {answer}")
        status_text.text("Updating history...")
        progress_bar.progress(75)
        query_time = datetime.now(timezone.utc)
        st.session_state.query_history.append({
            'package': selected_package,
            'query': query,
            'file': relevant_files,
            'answer': answer,
            'timestamp': query_time,
            'user': user
        })
        st.session_state['rate_limiter'].record_request()
        progress_bar.progress(100)
        status_text.text("Complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        return relevant_files, answer
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}", exc_info=True)
        progress_bar.empty()
        status_text.text(f"Error: {e}")
        raise e

@count_tokens_decorator
def perform_online_search(query: str) -> str:
    if not query or not isinstance(query, str):
        raise ValueError("Invalid query: must be a non-empty string.")
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except KeyError:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set. Please set it before running.")
    model = genai.GenerativeModel("models/gemini-2.5-flash-preview-04-17")
    response = model.generate_content(
        contents=query,
        tools="google_search_retrieval"
    )
    answer_text = ""
    if response.candidates:
        first_candidate = response.candidates[0]
        if first_candidate.content and first_candidate.content.parts:
            answer_text = first_candidate.content.parts[0].text
    return answer_text

###############################################################################
#                               CONFIGURATION
###############################################################################
def show_configuration(rag_system):
    with st.sidebar:
        st.header("Configuration ⚙️")
        with st.expander("Model Settings"):
            selected_package = st.selectbox(
                "Select Package",
                st.session_state['available_packages'],
                index=st.session_state['available_packages'].index(st.session_state['selected_package'])
            )
            st.session_state['selected_package'] = selected_package
            package_config = next(
                (pkg for pkg in rag_system.package_configs if pkg.name == selected_package),
                None
            )
            if package_config is None:
                st.error(f"Configuration for package '{selected_package}' not found.")
                return
            file_selection_model = st.selectbox(
                "File Selection Model",
                [
                    "qwen2.5-coder:3b",
                    "qwen2.5-coder:7b",
                    "gemini-2.5-pro-preview-03-25",
                    "gemini-2.5-flash-preview-04-17",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite"
                ],
                index=0
            )
            query_processing_model = st.selectbox(
                "Query Processing Model",
                [
                    "qwen2.5-coder:7b",
                    "qwen2.5-coder:3b",
                    "gemini-2.5-pro-preview-03-25",
                    "gemini-2.5-flash-preview-04-17",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite"
                ],
                index=0
            )
            rate_limit = st.slider(
                "Rate Limit (seconds)",
                min_value=1,
                max_value=30,
                value=st.session_state['config']['rate_limit']
            )
            current_user = st.text_input("Username", value=st.session_state['current_user'])
        with st.expander("Paper Checker Mode"):
            if st.session_state['config']['paper_checker_mode']:
                st.success("PaperCheckerMode is currently ENABLED.")
                if st.button("Disable PaperCheckerMode"):
                    st.session_state['config']['paper_checker_mode'] = False
                    ConfigManager.save_config(st.session_state['config'])
                    st.rerun()
            else:
                st.info("PaperCheckerMode is currently DISABLED.")
                if st.button("Enable PaperCheckerMode"):
                    st.session_state['config']['paper_checker_mode'] = True
                    ConfigManager.save_config(st.session_state['config'])
                    st.rerun()
        if st.button("Save Configuration"):
            st.session_state['config'].update({
                'file_selection_model': file_selection_model,
                'query_processing_model': query_processing_model,
                'rate_limit': rate_limit
            })
            st.session_state['current_user'] = current_user
            ConfigManager.save_config(st.session_state['config'])
            st.session_state['rate_limiter'] = RateLimiter(rate_limit)
            st.session_state['query_cache'] = QueryCache()
            st.success("Configuration saved successfully.")
            get_rag_system.clear()
            st.rerun()

###############################################################################
#                                 MAIN APP
###############################################################################
def main():
    show_header()
    show_system_status()
    display_health_status()
    rag_system = get_rag_system(st.session_state['config'])
    if rag_system is None:
        st.error("Failed to initialize RAG system.")
        return
    with st.sidebar:
        show_public_link_controls()
    show_configuration(rag_system)
    if st.button("Reset System"):
        reset_session_state()
        st.success("System has been reset.")
        st.rerun()
    if not st.session_state['ollama_ready']:
        if not check_ollama_server():
            st.error("❌ Ollama server is not running")
            if st.button("🚀 Start Ollama Server"):
                try:
                    subprocess.Popen(['ollama', 'serve'])
                    time.sleep(5)
                    if check_ollama_server():
                        st.session_state['ollama_ready'] = True
                        st.success("✅ Ollama server started successfully")
                        st.rerun()
                except FileNotFoundError:
                    st.error("❌ Ollama is not installed")
            return
        else:
            st.session_state['ollama_ready'] = True
    # --- Check whether to run Paper Checker Mode or Normal RAG Query Interface ---
    if st.session_state['config'].get('paper_checker_mode', False):
        # Paper Checker Mode is now handled externally via the paper_checker_mode module.
        paper_checker_interface()
    else:
        st.markdown("### Query Interface  🔍")
        selected_package = st.session_state['selected_package']
        query = st.text_area(
            "Enter your query for local docs (max 1000 chars):",
            height=100,
            placeholder="Enter your question about the selected package's documents..."
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            submit = st.button("🚀 Submit")
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.success("Query history cleared.")
                st.rerun()
        if submit and query:
            if not query.strip():
                st.error("Query cannot be empty.")
                return
            if len(query) > 1000:
                st.error("Query cannot exceed 1000 characters in normal mode. Please shorten your query.")
                return
            is_valid, error_message = QueryManager.validate_query(query)
            if not is_valid:
                st.error(error_message)
                return
            if not st.session_state['rate_limiter'].can_make_request():
                wait_time = st.session_state['rate_limiter'].time_until_next_request()
                st.warning(f"Please wait {wait_time:.1f} seconds before making another query.")
                return
            try:
                with st.spinner("Processing query locally..."):
                    relevant_files, answer = process_query_with_progress(
                        query,
                        rag_system,
                        selected_package,
                        user=st.session_state['current_user']
                    )
                    st.success(f"📄 Selected document(s): {relevant_files}")
                    st.markdown("### Local RAG Answer 💡")
                    st.markdown(answer)
            except ValueError as ve:
                st.error(f"Invalid query: {str(ve)}")
                logger.error(f"ValueError: {str(ve)}", exc_info=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error("Query processing error", exc_info=True)
        st.markdown("### Online Search 🌐")
        online_query = st.text_area(
            "Enter a query to search the web (max 1000 chars):",
            height=100,
            placeholder="Ask anything that might require live / web context..."
        )
        col_a, col_b = st.columns([1, 5])
        with col_a:
            online_submit = st.button("🔎 Search Online")
        with col_b:
            if st.button("🗑️ Clear Online Search History"):
                st.session_state['online_search_history'] = []
                st.success("Online search history cleared.")
                st.rerun()
        if online_submit and online_query:
            if not online_query.strip():
                st.error("Online query cannot be empty.")
            else:
                if len(online_query) > 1000:
                    st.error("Online query cannot exceed 1000 characters in normal mode. Please shorten your query.")
                    return
                if not st.session_state['rate_limiter'].can_make_request():
                    wait_time = st.session_state['rate_limiter'].time_until_next_request()
                    st.warning(f"Please wait {wait_time:.1f} seconds before making another query.")
                else:
                    with st.spinner("Searching online..."):
                        try:
                            online_answer = perform_online_search(online_query)
                            st.markdown("### Online Search Answer")
                            st.markdown(online_answer)
                            query_time = datetime.now(timezone.utc)
                            st.session_state['online_search_history'].append({
                                'query': online_query,
                                'answer': online_answer,
                                'timestamp': query_time,
                                'user': st.session_state['current_user']
                            })
                            st.session_state['rate_limiter'].record_request()
                        except Exception as e:
                            st.error(f"An error occurred while searching online: {str(e)}")
                            logger.error("Online search error", exc_info=True)
        show_query_history()
        show_online_search_history()
    total_input, total_output = global_token_counter.get_total_tokens()
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Token Usage 📊")
        st.metric("Total Input Tokens", total_input)
        st.metric("Total Output Tokens", total_output)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
# --- END OF FILE app.py ---