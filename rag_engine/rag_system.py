import logging
import sys
import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
from prometheus_client import Counter, Histogram, Gauge
import tenacity
import chromadb
from collections import OrderedDict
from logging.handlers import RotatingFileHandler

# Custom Logger Class
class RAGLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Add rotating file handler
        handler = RotatingFileHandler('rag_system.log', maxBytes=10485760, backupCount=5)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Add stream handler for console output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

# Initialize logger
logger = RAGLogger(__name__)

@dataclass
class PerformanceMetrics:
    query_counter: Counter = Counter('rag_queries_total', 'Total number of queries processed')
    query_latency: Histogram = Histogram('rag_query_duration_seconds', 'Query processing duration')
    cache_hits: Counter = Counter('rag_cache_hits_total', 'Number of cache hits')
    model_calls: Dict[str, Counter] = field(default_factory=dict)
    memory_usage: Gauge = Gauge('rag_memory_usage_bytes', 'Memory usage in bytes')
    request_duration: Histogram = field(
        default_factory=lambda: Histogram(
            'rag_request_duration_seconds',
            'Request duration in seconds',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
        )
    )

    def record_query(self, duration: float):
        self.query_counter.inc()
        self.query_latency.observe(duration)

    def record_cache_hit(self):
        self.cache_hits.inc()

    def record_model_call(self, model_name: str):
        try:
            # Sanitize the model name for Prometheus compatibility
            sanitized_name = model_name.replace('.', '_').replace(':', '_').replace('-', '_')
            metric_name = f'rag_model_calls_{sanitized_name}'

            if model_name not in self.model_calls:
                self.model_calls[model_name] = Counter(
                    metric_name,
                    f'Number of calls to model {model_name}'
                )
            self.model_calls[model_name].inc()

        except ValueError as ve:
            logger.error(f"Invalid metric name creation: {str(ve)}")
            # Create a fallback metric with a generic name
            fallback_name = f"rag_model_calls_model_{len(self.model_calls)}"
            self.model_calls[model_name] = Counter(
                fallback_name,
                f'Number of calls to model (fallback counter)'
            )
            self.model_calls[model_name].inc()
        except Exception as e:
            logger.error(f"Unexpected error in record_model_call: {str(e)}")
            # Don't let metric recording failures affect the main application flow
            pass

    def record_memory_usage(self):
        import psutil
        process = psutil.Process(os.getpid())
        self.memory_usage.set(process.memory_info().rss)

    def record_request_time(self, duration: float):
        self.request_duration.observe(duration)

# TTL Cache Class
class TTLCache(OrderedDict):
    def __init__(self, maxsize=1000, ttl=3600):
        super().__init__()
        self.maxsize = maxsize
        self.ttl = ttl

    def __getitem__(self, key):
        value, timestamp = super().__getitem__(key)
        if time.time() - timestamp > self.ttl:
            del self[key]
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, (value, time.time()))
        if len(self) > self.maxsize:
            self.popitem(last=False)

class RAGSystem:
    def __init__(self, json_directory: str, kbi_path: str):
        self.json_directory = json_directory
        self.kbi_path = kbi_path
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.cache = TTLCache()
        self.ollama_session = requests.Session()
        self.metrics = PerformanceMetrics()
        self.models = {
            'file_selection': 'qwen2.5-coder:3b',
            'query_processing': 'qwen2.5-coder:7b'
        }

        # Add persistent directory
        self.persist_directory = os.path.join(os.getcwd(), "chroma_db")
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize Chroma client settings
        self.chroma_settings = chromadb.config.Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=self.persist_directory
        )

        # Initialize Chroma client with connection pooling
        self.chroma_client = chromadb.Client(self.chroma_settings)

        # Initialize connection pool for Ollama
        self.ollama_session.mount(
            'http://',
            requests.adapters.HTTPAdapter(
                max_retries=3,
                pool_connections=10,
                pool_maxsize=10
            )
        )

        self.kbi_vectorstore = self.create_kbi_vectorstore()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    @lru_cache(maxsize=100)
    def get_file_embeddings(self, file_path):
        """Cache embeddings for frequently accessed files"""
        try:
            with open(file_path, 'r') as file:
                file_data = [{"content": file.read(), "source": file_path}]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            file_splits = text_splitter.create_documents(
                texts=[doc["content"] for doc in file_data],
                metadatas=[{"source": doc["source"]} for doc in file_data]
            )

            embeddings = GPT4AllEmbeddings().embed_documents([doc.page_content for doc in file_splits])
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings for {file_path}: {str(e)}")
            return []

    def batch_embed_documents(self, documents, batch_size=32):
        """Generate embeddings in batches"""
        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_embeddings = self.get_file_embeddings(batch)
            embeddings.extend(batch_embeddings)
        return embeddings

    async def batch_process_queries(self, queries):
        """Process multiple queries in parallel"""
        tasks = [self.process_query(q) for q in queries]
        return await asyncio.gather(*tasks)

    def check_ollama_status(self):
        """Check if Ollama is running and required models are available"""
        try:
            # Check if Ollama server is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                return False, "Ollama server is not running"

            # Check for required models
            models = response.json().get("models", [])
            required_models = list(self.models.values())
            logger.info(f"Available models: {[m.get('name', '') for m in models]}")
            logger.info(f"Required models: {required_models}")

            missing_models = [model for model in required_models
                              if not any(m.get("name") == model for m in models)]

            if missing_models:
                return False, f"Missing required models: {', '.join(missing_models)}"

            return True, "Ollama is ready"
        except requests.ConnectionError:
            return False, "Cannot connect to Ollama server"
        except requests.exceptions.Timeout:
            return False, "Ollama server connection timed out"
        except Exception as e:
            return False, f"An unexpected error occurred: {str(e)}"

    def validate_json_file(self, file_path):
        """Validate a JSON file"""
        try:
            with open(file_path, 'r') as file:
                json.load(file)
                logger.info(f"✓ {file_path} is valid JSON")
                return True
        except json.JSONDecodeError as e:
            logger.error(f"Error in file {file_path}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return False

    def check_all_json_files(self):
        """Check all JSON files in the directory"""
        logger.info(f"Checking JSON files in {self.json_directory}")
        all_valid = True
        for filename in os.listdir(self.json_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(self.json_directory, filename)
                if not self.validate_json_file(file_path):
                    all_valid = False
        return all_valid

    def create_kbi_vectorstore(self, persistence_dir="./chroma_db"):
        try:
            # Load and validate KBI data
            with open(self.kbi_path, 'r') as file:
                kbi_data = json.load(file)
                logger.info(f"Successfully loaded KBI data from {self.kbi_path}")

            if not isinstance(kbi_data, dict) or 'files' not in kbi_data:
                raise ValueError("Invalid KBI data structure")

            # Process documents
            kbi_docs = []
            for file_info in kbi_data.get('files', []):
                try:
                    if not all(key in file_info for key in ['name', 'introduction']):
                        logger.warning(f"Skipping incomplete file info: {file_info}")
                        continue

                    text = f"File: {file_info['name']}\nIntroduction: {file_info['introduction']}"
                    kbi_docs.append({"content": text, "source": "KBI.json"})
                except Exception as doc_error:
                    logger.error(f"Error processing document: {str(doc_error)}")
                    continue

            if not kbi_docs:
                raise ValueError("No valid documents found in KBI data")

            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )

            # Create splits from the documents
            kbi_splits = text_splitter.create_documents(
                texts=[doc["content"] for doc in kbi_docs],
                metadatas=[{"source": doc["source"]} for doc in kbi_docs]
            )

            if not kbi_splits:
                raise ValueError("Text splitting produced no documents")

            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=kbi_splits,  # Now kbi_splits is properly defined
                embedding=GPT4AllEmbeddings(),
                persist_directory=persistence_dir,
                collection_name="kbi_collection",
                client=self.chroma_client
            )

            logger.info(f"Successfully created vector store with {len(kbi_splits)} chunks")
            return vectorstore

        except FileNotFoundError:
            logger.error(f"KBI file not found at {self.kbi_path}")
            raise
        except json.JSONDecodeError as je:
            logger.error(f"Invalid JSON in KBI file: {str(je)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in create_kbi_vectorstore: {str(e)}")
            raise

    def find_relevant_file(self, query):
        """Find the most relevant file for a given query"""
        start_time = time.time()
        try:
            # Check Ollama status first
            status, message = self.check_ollama_status()
            if not status:
                raise Exception(f"Ollama is not ready: {message}")

            if query in self.cache:
                self.metrics.record_cache_hit()
                cached_result = self.cache[query]
                logger.info(f"Cache hit for query: {query}")
                return cached_result

            file_template = """Based on the following context and question, determine which JSON file would be most relevant.
            Return ONLY the filename, nothing else.
            Context: {context}
            Question: {question}
            Filename:"""

            file_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=file_template,
            )

            logger.info(f"Using model {self.models['file_selection']} for file selection")
            self.metrics.record_model_call(self.models['file_selection'])
            llm = Ollama(
                model=self.models['file_selection'],
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )

            file_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=self.kbi_vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": file_prompt},
            )

            result = file_chain({"query": query})["result"].strip()
            self.cache[query] = result
            logger.info(f"Found relevant file: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in find_relevant_file: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            self.metrics.record_request_time(duration)

    def process_query(self, query, relevant_file):
        """Process a query using the relevant file"""
        start_time = time.time()
        try:
            # Check Ollama status first
            status, message = self.check_ollama_status()
            if not status:
                raise Exception(f"Ollama is not ready: {message}")

            file_path = os.path.join(self.json_directory, relevant_file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {relevant_file} not found")

            with open(file_path, 'r') as file:
                file_data = [{"content": file.read(), "source": relevant_file}]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            file_splits = text_splitter.create_documents(
                texts=[doc["content"] for doc in file_data],
                metadatas=[{"source": doc["source"]} for doc in file_data]
            )

            collection_name = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            file_vectorstore = Chroma.from_documents(
                documents=file_splits,
                embedding=GPT4AllEmbeddings(),
                collection_name=collection_name,
                client=self.chroma_client  # Pass the client instance
            )

            template = """Use the following context to answer the question about the JSON data.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            You should always using omicverse in python to provide the answer.
            You are focus on the code not the specific gene and disease in the JSON file.
            Use Step by Step with code and keep the answer as concise as possible. 
            Context: {context}
            Question: {question}
            Helpful Answer:"""

            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )

            logger.info(f"Using model {self.models['query_processing']} for query processing")
            self.metrics.record_model_call(self.models['query_processing'])
            llm = Ollama(
                model=self.models['query_processing'],
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=file_vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": qa_prompt},
            )

            logger.info("Generating answer...")
            result = qa_chain({"query": query})

            self._cleanup_old_collections()

            return result["result"]

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            self.metrics.record_query(duration)
            self.metrics.record_memory_usage()
            self.metrics.record_request_time(duration)

    def list_json_files(self):
        """List all JSON files in the directory"""
        return [f for f in os.listdir(self.json_directory) if f.endswith('.json')]

    def get_system_health(self) -> Dict:
        """Get system health metrics"""
        try:
            total_queries = float(self.metrics.query_counter._value.get())
            cache_hits = float(self.metrics.cache_hits._value.get())
            query_latency_sum = float(self.metrics.query_latency.sum._value.get())

            return {
                'cache_size': len(self.cache),
                'cache_hits': int(cache_hits),
                'total_queries': int(total_queries),
                'avg_latency': round(query_latency_sum / total_queries, 2) if total_queries > 0 else 0.00,
                'model_usage': {
                    model: int(counter._value.get())
                    for model, counter in self.metrics.model_calls.items()
                },
                'memory_usage': self.get_memory_usage(),
                'ollama_status': self.check_ollama_status()
            }
        except Exception as e:
            logger.error(f"Error getting system health metrics: {str(e)}")
            return {
                'cache_size': 0,
                'cache_hits': 0,
                'total_queries': 0,
                'avg_latency': 0.00,
                'model_usage': {},
                'memory_usage': 0,
                'ollama_status': ("Unknown", str(e))
            }

    def get_memory_usage(self):
        """Get current memory usage"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def get_cache_health(self):
        """Get cache health metrics"""
        return {
            'size': len(self.cache),
            'maxsize': self.cache.maxsize,
            'ttl': self.cache.ttl,
            'hits': int(self.metrics.cache_hits._value.get())
        }

    def check_vectorstore_health(self):
        """Check vector store health"""
        try:
            self.chroma_client.heartbeat()
            return "OK"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_detailed_health(self):
        """Get detailed system health"""
        return {
            'system_status': self.check_ollama_status(),
            'cache_status': self.get_cache_health(),
            'vectorstore_status': self.check_vectorstore_health(),
            'memory_usage': self.get_memory_usage()
        }

    def _cleanup_old_collections(self):
        """Clean up old vector store collections"""
        try:
            current_time = datetime.now()
            collections = self.chroma_client.list_collections()

            for collection in collections:
                if collection.name.startswith('query_'):
                    collection_time_str = collection.name.split('_')[1]
                    try:
                        collection_time = datetime.strptime(collection_time_str, '%Y%m%d_%H%M%S')
                        if (current_time - collection_time).total_seconds() > 3600:  # 1 hour
                            self.chroma_client.delete_collection(collection.name)
                            logger.info(f"Deleted old collection: {collection.name}")
                    except ValueError:
                        logger.warning(f"Failed to parse timestamp from collection name: {collection.name}")
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(requests.ConnectionError)
    )
    def _call_ollama(self, endpoint: str, data: Dict) -> Dict:
        """Resilient Ollama API calls with retry logic"""
        try:
            response = self.ollama_session.post(
                f"http://localhost:11434/api/{endpoint}",
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Ollama API timeout for endpoint {endpoint}")
            raise TimeoutError("Ollama API request timed out")
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"Connection error to Ollama API: {str(ce)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API call failed: {str(e)}")
            raise
        except json.JSONDecodeError as je:
            logger.error(f"Failed to decode Ollama API response: {str(je)}")
            raise ValueError("Invalid JSON response from Ollama API")

    def cleanup(self):
        """Cleanup method to handle resources properly"""
        try:
            # Clean up vector stores
            if hasattr(self, 'kbi_vectorstore') and self.kbi_vectorstore is not None:
                self.kbi_vectorstore._client.reset()

            # Clean up Chroma client
            self.chroma_client.reset()

            # Close Ollama session
            self.ollama_session.close()

            # Shutdown thread pool executor
            self.executor.shutdown()

            # Remove persistent directory
            if os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory, ignore_errors=True)

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")