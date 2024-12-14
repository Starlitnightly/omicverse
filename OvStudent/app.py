# --- START OF FILE app.py ---
import streamlit as st
import json
from datetime import datetime, timezone
import os
import subprocess
import time
import requests
import getpass
import psutil
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from collections import OrderedDict

from rag_system import RAGSystem
from config_manager import ConfigManager
from system_monitor import SystemMonitor
from rate_limiter import RateLimiter
from query_cache import QueryCache
from query_manager import QueryManager


# Set up logging with rotating file handler
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    handler = RotatingFileHandler(
        log_dir / 'streamlit_app.log',
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            handler
        ]
    )


setup_logging()


# Initialize session state
def initialize_session_state():
    default_state = {
        'ollama_ready': False,
        'models_installed': False,
        'query_history': [],
        'rate_limiter': None,
        'query_cache': None,
        'config': {
            'file_selection_model': 'qwen2.5-coder:3b',
            'query_processing_model': 'qwen2.5-coder:7b',
            'rate_limit': 5,  # seconds between queries
            # Add your default directories here
            'converted_jsons_directory': "/Users/kq_m3m/PycharmProjects/OVMaster/Converted_Jsons",
            'annotated_scripts_directory': "/Users/kq_m3m/PycharmProjects/OVMaster/Converted_Scripts_Annotated"
        },
        'current_user': getpass.getuser()  # Get the current username
    }

    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Initialize RateLimiter if not already set
    if st.session_state['rate_limiter'] is None:
        st.session_state['rate_limiter'] = RateLimiter(st.session_state['config']['rate_limit'])

    # Initialize QueryCache if not already set
    if st.session_state['query_cache'] is None:
        st.session_state['query_cache'] = QueryCache()


initialize_session_state()


# Cache for RAGSystem using @st.cache_resource to ensure singleton
@st.cache_resource
def get_rag_system(converted_jsons_directory, annotated_scripts_directory):
    try:
        return RAGSystem(converted_jsons_directory, annotated_scripts_directory)
    except Exception as e:
        logging.error(f"Failed to initialize RAG system: {str(e)}")
        return None


# Function to display the header
def show_header():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("Agentic OmicVerse 🧬")
    with col2:
        # Display current time using a placeholder that will update automatically
        current_time = datetime.now(timezone.utc)  # Get the current time in UTC
        st.info(f"📅 UTC: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    with col3:
        # Display the current username
        st.info(f"👤 User: {st.session_state['current_user']}")


# Function to display system status
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


# Function to check if Ollama server is running
def check_ollama_server() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# Function to display health status
def display_health_status():
    healthy, checks = check_system_health()
    with st.sidebar:
        st.header("System Health ✅" if healthy else "System Health ⚠️")
        for component, status in checks.items():
            if status:
                st.success(f"{component} is running")
            else:
                st.error(f"{component} is not running")


# Function to perform health checks
def check_system_health():
    health_checks = {
        'Ollama Server': check_ollama_server(),
    }
    all_healthy = all(health_checks.values())
    return all_healthy, health_checks


# Function to display configuration settings
def show_configuration():
    with st.sidebar:
        st.header("Configuration ⚙️")
        with st.expander("Model Settings"):
            file_selection_model = st.selectbox(
                "File Selection Model",
                ["qwen2.5-coder:3b", "qwen2.5-coder:7b", "gemini-pro", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp"],
                index=["qwen2.5-coder:3b", "qwen2.5-coder:7b", "gemini-pro", "gemini-1.5-flash-8b",
                       "gemini-2.0-flash-exp"].index(
                    st.session_state['config'].get('file_selection_model', "qwen2.5-coder:3b")
                )
            )
            query_processing_model = st.selectbox(
                "Query Processing Model",
                ["qwen2.5-coder:7b", "qwen2.5-coder:3b", "gemini-pro", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp"],
                index=["qwen2.5-coder:7b", "qwen2.5-coder:3b", "gemini-pro", "gemini-1.5-flash-8b",
                       "gemini-2.0-flash-exp"].index(
                    st.session_state['config'].get('query_processing_model', "qwen2.5-coder:7b")
                )
            )

            # If using Gemini, request the API key (optional if not needed anymore)
            # gemini_api_key = None  # The redesigned rag_system.py doesn't require this

            rate_limit = st.slider(
                "Rate Limit (seconds)",
                min_value=1,
                max_value=30,
                value=st.session_state['config']['rate_limit']
            )
            current_user = st.text_input("Username", value=st.session_state['current_user'])

            # Directories
            converted_jsons_directory = st.text_input("Converted JSONs Directory",
                                                      value=st.session_state['config']['converted_jsons_directory'])
            annotated_scripts_directory = st.text_input("Annotated Scripts Directory",
                                                        value=st.session_state['config']['annotated_scripts_directory'])

            if st.button("Save Configuration"):
                st.session_state['config'].update({
                    'file_selection_model': file_selection_model,
                    'query_processing_model': query_processing_model,
                    'rate_limit': rate_limit,
                    'converted_jsons_directory': converted_jsons_directory,
                    'annotated_scripts_directory': annotated_scripts_directory
                })
                st.session_state['current_user'] = current_user
                ConfigManager.save_config(st.session_state['config'])
                st.session_state['rate_limiter'] = RateLimiter(rate_limit)
                st.session_state['query_cache'] = QueryCache()
                st.success("Configuration saved successfully.")


# Function to process query with progress tracking using the new RAG logic
def process_query_with_progress(query, rag_system):
    if not query or not isinstance(query, str):
        raise ValueError("Invalid query: Query must be a non-empty string")

    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        logging.info(f"Processing query: {query}")
        logging.info(f"RAG System State: {rag_system is not None}")

        status_text.text("Finding relevant documents...")
        progress_bar.progress(25)

        # Pass query directly as a string
        relevant_files = rag_system.find_relevant_files(query)

        status_text.text("Generating answer from annotated scripts...")
        progress_bar.progress(50)
        answer = rag_system.answer_query_with_annotated_scripts(query, relevant_files)
        logging.info(f"Answer: {answer}")

        status_text.text("Updating history...")
        progress_bar.progress(75)

        query_time = datetime.now(timezone.utc)
        st.session_state.query_history.append({
            'query': query,
            'file': relevant_files,
            'answer': answer,
            'timestamp': query_time,
            'user': st.session_state['current_user']
        })

        st.session_state['rate_limiter'].record_request()
        progress_bar.progress(100)
        status_text.text("Complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        return relevant_files, answer
    except Exception as e:
        logging.error(f"Query processing error: {str(e)}", exc_info=True)
        progress_bar.empty()
        status_text.text(f"Error: {e}")
        raise e


# Function to display query history
def show_query_history():
    with st.sidebar:
        st.header("Query History 📜")
        for idx, item in enumerate(reversed(st.session_state.query_history[-10:])):
            with st.expander(f"Query {len(st.session_state.query_history) - idx}: {item['query'][:30]}..."):
                st.markdown(f"**Time:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
                st.markdown(f"**User:** {item['user']}")
                st.markdown(f"**Document(s):** {item['file']}")
                st.markdown(f"**Answer:** {item['answer']}")
                st.markdown("---")


# Main function
def main():
    show_header()
    show_system_status()
    display_health_status()
    show_configuration()

    if st.button("Reset System"):
        st.session_state.query_history = []
        st.session_state['rate_limiter'] = RateLimiter(st.session_state['config']['rate_limit'])
        st.session_state['query_cache'] = QueryCache()
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

    # Initialize RAGSystem via cached function
    converted_jsons_directory = st.session_state['config']['converted_jsons_directory']
    annotated_scripts_directory = st.session_state['config']['annotated_scripts_directory']
    rag_system = get_rag_system(converted_jsons_directory, annotated_scripts_directory)

    if rag_system is None:
        st.error("Failed to initialize RAG system.")
        return

    st.markdown("### Query Interface 🔍")
    query = st.text_area(
        "Enter your query:",
        height=100,
        placeholder="Enter your question about the documents..."
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🚀 Submit")
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()

    if submit and query:
        # Add validation to ensure query is not empty or just whitespace
        if not query.strip():
            st.error("Query cannot be empty. Please enter a valid query.")
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
            with st.spinner("Processing query..."):
                # Log the query and state before processing
                logging.info(f"Processing query: {query!r}")
                logging.info(f"RAG system state: initialized={rag_system is not None}")

                relevant_files, answer = process_query_with_progress(query, rag_system)
                st.success(f"📄 Selected document(s): {relevant_files}")
                st.markdown("### Answer 💡")
                st.markdown(answer)
        except ValueError as ve:
            st.error(f"Invalid query: {str(ve)}")
            logging.error(f"ValueError in query processing: {str(ve)}", exc_info=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error("Query processing error", exc_info=True)

    show_query_history()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
# --- END OF FILE app.py ---