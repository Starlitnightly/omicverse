import streamlit as st
import json
from datetime import datetime, timezone, timedelta
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
# Import the RAGSystem
from rag_system import RAGSystem, RAGLogger

# Set up logging with rotating file handler
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    handler = RotatingFileHandler(
        log_dir / 'streamlit_app.log',
        maxBytes=10*1024*1024,  # 10 MB
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
        },
        'current_time': datetime(2024, 12, 8, 13, 19, 36, tzinfo=timezone.utc),
        'current_user': 'HendricksJudy'
    }

    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Cache for RAGSystem
@st.cache_resource
def get_rag_system():
    try:
        json_directory = os.path.join(os.path.dirname(__file__), "ovrawmjson")
        kbi_path = os.path.join(json_directory, "KBI.json")
        return RAGSystem(json_directory, kbi_path)
    except Exception as e:
        logging.error(f"Failed to initialize RAG system: {str(e)}")
        return None

# System Monitor class with enhanced metrics
class SystemMonitor:
    @staticmethod
    def get_system_stats():
        process = psutil.Process()
        memory = psutil.virtual_memory()
        return {
            'memory_usage': process.memory_info().rss / 1024 / 1024,  # MB
            'cpu_percent': psutil.cpu_percent(interval=1),
            'uptime': time.time() - process.create_time(),
            'system_memory': {
                'total': memory.total / (1024 ** 3),        # GB
                'available': memory.available / (1024 ** 3),  # GB
                'percent': memory.percent
            }
        }

    @staticmethod
    def format_uptime(seconds):
        return str(timedelta(seconds=int(seconds)))

# RateLimiter class for query rate limiting
class RateLimiter:
    def __init__(self, limit_seconds):
        self.limit_seconds = limit_seconds
        self.last_request_time = None

    def can_make_request(self):
        if not self.last_request_time:
            return True
        time_since_last = time.time() - self.last_request_time
        return time_since_last >= self.limit_seconds

    def time_until_next_request(self):
        if not self.last_request_time:
            return 0
        time_since_last = time.time() - self.last_request_time
        return max(0, self.limit_seconds - time_since_last)

    def record_request(self):
        self.last_request_time = time.time()

# Initialize RateLimiter
if st.session_state['rate_limiter'] is None:
    st.session_state['rate_limiter'] = RateLimiter(st.session_state['config']['rate_limit'])

# QueryCache class for cache management
class QueryCache:
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

# Initialize QueryCache
if st.session_state['query_cache'] is None:
    st.session_state['query_cache'] = QueryCache()

# ConfigManager class for configuration management
class ConfigManager:
    CONFIG_PATH = Path('config.json')

    @staticmethod
    def load_config():
        if ConfigManager.CONFIG_PATH.exists():
            with open(ConfigManager.CONFIG_PATH, 'r') as f:
                return json.load(f)
        else:
            return st.session_state['config']

    @staticmethod
    def save_config(config):
        with open(ConfigManager.CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)

# Load configuration
st.session_state['config'] = ConfigManager.load_config()

# Function to display the header
def show_header():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("Agentic OmicVerse 🧬")
    with col2:
        # Using the specified datetime
        st.info(f"📅 UTC: {datetime(2024, 12, 8, 13, 20, 42, tzinfo=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    with col3:
        # Using the specified username
        st.info(f"👤 User: HendricksJudy")

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
                ["qwen2.5-coder:3b", "qwen2.5-coder:7b"],
                index=["qwen2.5-coder:3b", "qwen2.5-coder:7b"].index(
                    st.session_state['config']['file_selection_model']
                )
            )
            query_processing_model = st.selectbox(
                "Query Processing Model",
                ["qwen2.5-coder:7b", "qwen2.5-coder:3b"],
                index=["qwen2.5-coder:7b", "qwen2.5-coder:3b"].index(
                    st.session_state['config']['query_processing_model']
                )
            )
            rate_limit = st.slider(
                "Rate Limit (seconds)",
                min_value=1,
                max_value=30,
                value=st.session_state['config']['rate_limit']
            )

            if st.button("Save Configuration"):
                st.session_state['config'].update({
                    'file_selection_model': file_selection_model,
                    'query_processing_model': query_processing_model,
                    'rate_limit': rate_limit
                })
                ConfigManager.save_config(st.session_state['config'])
                st.session_state['rate_limiter'] = RateLimiter(rate_limit)
                st.success("Configuration saved successfully.")


# Function to process query with progress tracking
def process_query_with_progress(query, rag_system):
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        status_text.text("Finding relevant document...")
        progress_bar.progress(25)
        relevant_file = rag_system.find_relevant_file(query)
        status_text.text("Processing query...")
        progress_bar.progress(50)
        answer = rag_system.process_query(query, relevant_file)
        status_text.text("Updating history...")
        progress_bar.progress(75)

        # Using the specified datetime for query history
        query_time = datetime(2024, 12, 8, 13, 21, 29, tzinfo=timezone.utc)
        st.session_state.query_history.append({
            'query': query,
            'file': relevant_file,
            'answer': answer,
            'timestamp': query_time,
            'user': 'HendricksJudy'
        })

        st.session_state['rate_limiter'].record_request()
        progress_bar.progress(100)
        status_text.text("Complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        return relevant_file, answer
    except Exception as e:
        logging.error(f"Query processing error: {str(e)}")
        progress_bar.empty()
        status_text.text(f"Error: {e}")
        raise e


# QueryManager class
class QueryManager:
    @staticmethod
    def validate_query(query):
        if not query or len(query.strip()) < 3:
            return False, "Query must be at least 3 characters long"
        if len(query) > 1000:
            return False, "Query must be less than 1000 characters"
        return True, ""


# Function to display query history
def show_query_history():
    with st.sidebar:
        st.header("Query History 📜")
        for idx, item in enumerate(reversed(st.session_state.query_history[-10:])):
            with st.expander(f"Query {len(st.session_state.query_history) - idx}: {item['query'][:30]}..."):
                st.markdown(f"**Time:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
                st.markdown(f"**User:** {item['user']}")
                st.markdown(f"**Document:** {item['file']}")
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

    rag_system = get_rag_system()
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
                relevant_file, answer = process_query_with_progress(query, rag_system)
                st.success(f"📄 Selected document: {relevant_file}")
                st.markdown("### Answer 💡")
                st.markdown(answer)
        except Exception as e:
            logging.error(f"Query processing error: {str(e)}")
            st.error(f"Error processing query: {str(e)}")

    show_query_history()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")