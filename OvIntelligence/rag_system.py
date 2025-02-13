# --- START OF FILE rag_system.py ---
import os
import time
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import logging
from logging.handlers import RotatingFileHandler

import chromadb
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Use GPT4AllEmbeddings in all cases for file selection and query processing
from langchain_community.embeddings import GPT4AllEmbeddings
# For fallback LLM when not using Gemini
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# NEW IMPORTS FOR GEMINI LLM
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from dataclasses import dataclass

# Import the global token counter
from token_counter import global_token_counter

# Import the model selection helper from our independent script
from model_selector import get_llm

# --------------------------------------------------
# Configure Gemini model if API key is available.
# It is recommended to set the GEMINI_API_KEY in your environment variables.
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not set. Gemini-based functions may not work properly.")

# --------------------------------------------------
def refine_code_with_gemini(baseline_code: str, user_query: str) -> str:
    """
    Given a baseline code snippet and the user query context, use the Gemini Stage2 model
    to refine the code. The refined code should be more precise, robust, and production-ready.
    """
    prompt = f"""
You are an expert single-cell bioinformatics code generator.

Given the baseline code snippet below and the user query, refine the code to be more precise,
robust, and production-ready. Ensure the code adheres to PEP8 standards, includes proper error handling,
and optimizes the solution as needed.

User Query: {user_query}

Baseline Code:
{baseline_code}

Refined Code:"""
    try:
        # Instead of using genai.models.generate_content (which no longer exists),
        # we now create a GenerativeModel instance and call its generate_content() method.
        model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            refined_code = response.text.strip()
        else:
            refined_code = baseline_code  # Fallback to baseline if no response text
    except Exception as e:
        logging.getLogger(__name__).error(f"Error during Gemini refinement: {e}", exc_info=True)
        refined_code = baseline_code  # Return the baseline code as a fallback
    return refined_code

# --------------------------------------------------
def setup_logging():
    """
    Sets up logging with both file and console handlers.
    Logs are stored in the 'logs' directory with rotation to prevent oversized files.
    """
    logger = logging.getLogger('rag_system_code_optimized')
    logger.setLevel(logging.DEBUG)

    os.makedirs('logs', exist_ok=True)

    file_handler = RotatingFileHandler(
        'logs/rag_system_code_optimized.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

@dataclass
class PackageConfig:
    """
    Data class to encapsulate configuration details for each package.
    """
    name: str
    converted_jsons_directory: str
    annotated_scripts_directory: str
    file_selection_model: str
    query_processing_model: str

class CodeAwareTextSplitter(CharacterTextSplitter):
    """
    A custom text splitter that attempts to split code more gracefully.
    Currently uses line-based splitting but can be enhanced to split on function/class boundaries.
    """
    def __init__(self, chunk_size=2000, chunk_overlap=200):
        super().__init__(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

class FirstStageRAG:
    """
    First stage of the RAG system responsible for retrieving relevant file descriptions.
    """
    def __init__(
            self,
            converted_jsons_directory: str,
            persist_dir: str,
            file_selection_model: str,
            chroma_client: chromadb.Client,
            top_k_files: int = 5,
            package_name: str = "default"
    ):
        self.package_name = package_name
        self.converted_jsons_directory = converted_jsons_directory
        self.persist_dir = persist_dir
        self.file_selection_model = file_selection_model
        self.chroma_client = chroma_client
        self.top_k_files = top_k_files

        logger.info(f"Initializing FirstStageRAG for package '{self.package_name}'...")

        self.collection_name = f"{self.package_name}_file_descriptions"
        self.collection = self._load_or_create_collection()

        if self.collection.count() == 0:
            logger.info(f"Populating first-stage collection for package '{self.package_name}' with file descriptions...")
            self._index_file_descriptions()
            logger.info("File descriptions indexed successfully.")

        logger.info(f"FirstStageRAG for package '{self.package_name}' initialized successfully.")

    def _load_or_create_collection(self):
        """
        Attempts to load an existing ChromaDB collection. If it doesn't exist, creates a new one.
        """
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Successfully loaded collection '{self.collection_name}'.")
            return collection
        except chromadb.errors.InvalidCollectionException as e:
            logger.warning(f"Collection '{self.collection_name}' not found: {str(e)}. Creating new collection.")
            collection = self.chroma_client.create_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' created successfully.")
            return collection
        except Exception as e:
            logger.error(f"Unexpected error while accessing collection '{self.collection_name}': {str(e)}", exc_info=True)
            raise

    def _index_file_descriptions(self):
        """
        Indexes file descriptions from JSON files into the ChromaDB collection.
        """
        embeddings = GPT4AllEmbeddings(model=self.file_selection_model)
        docs = []
        metadatas = []
        ids = []

        for root, _, files in os.walk(self.converted_jsons_directory):
            for fname in files:
                if fname.endswith(".json"):
                    fpath = os.path.join(root, fname)
                    with open(fpath, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            description = data.get("description", "")
                            file_name = data.get("file", fname)
                            if not description:
                                logger.warning(f"JSON file '{fpath}' has no 'description'. Skipping.")
                                continue
                            docs.append(description)
                            metadatas.append({"file": file_name})
                            ids.append(file_name)
                        except Exception as e:
                            logger.error(f"Error reading {fpath}: {str(e)}", exc_info=True)
        if docs:
            doc_embeddings = embeddings.embed_documents(docs)
            self.collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=doc_embeddings)
            logger.info(f"Indexed {len(docs)} documents into collection '{self.collection_name}'.")
        else:
            logger.warning(f"No documents to index in FirstStageRAG for package '{self.package_name}'.")

    def find_relevant_files(self, query: Union[str, Dict[str, Any]]) -> List[str]:
        """
        Finds the top_k_files most relevant files based on the query.
        """
        query_text = query.get("query", "") if isinstance(query, dict) else query
        if not query_text.strip():
            logger.warning("Empty query received in FirstStageRAG.")
            return []
        embeddings = GPT4AllEmbeddings(model=self.file_selection_model)
        query_embedding = embeddings.embed_query(query_text)
        try:
            results = self.collection.query(query_embeddings=[query_embedding], n_results=self.top_k_files)
            logger.info(f"Query executed successfully on collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error querying collection '{self.collection_name}': {str(e)}", exc_info=True)
            return []
        matched_files = []
        if results and "metadatas" in results and results["metadatas"]:
            for meta in results["metadatas"][0]:
                file_name = meta.get("file", None)
                if file_name:
                    matched_files.append(file_name)
        logger.info(f"Top {self.top_k_files} matched files for query '{query_text}' in package '{self.package_name}': {matched_files}")
        return matched_files

class SecondStageRAG:
    """
    Second stage of the RAG system responsible for generating answers based on annotated scripts.
    """
    def __init__(
            self,
            annotated_scripts_directory: str,
            query_processing_model: str,
            chroma_client: chromadb.Client,
            code_chunk_size: int = 2000,
            code_chunk_overlap: int = 200,
            top_k_chunks: int = 3,
            package_name: str = "default"
    ):
        self.package_name = package_name
        self.annotated_scripts_directory = annotated_scripts_directory
        self.query_processing_model = query_processing_model
        self.chroma_client = chromadb.Client(
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=""
            )
        ) if chroma_client is None else chroma_client
        self.code_chunk_size = code_chunk_size
        self.code_chunk_overlap = code_chunk_overlap
        self.top_k_chunks = top_k_chunks

        logger.info(f"Initializing SecondStageRAG for package '{self.package_name}'...")
        logger.info(f"SecondStageRAG for package '{self.package_name}' initialized successfully.")

    def _find_file_path(self, file_name: str) -> Optional[str]:
        """
        Recursively search for a file within the annotated_scripts_directory.
        """
        for root, dirs, files in os.walk(self.annotated_scripts_directory):
            if file_name in files:
                return os.path.join(root, file_name)
        return None

    def answer_query_with_annotated_scripts(self, query: str, relevant_files: List[str]) -> str:
        """
        Generates an answer based on the relevant annotated scripts for a given package and query.
        """
        logger.info(f"Starting second stage with query: '{query}' and relevant files: {relevant_files}")
        if not relevant_files:
            logger.info(f"No relevant files provided to second stage for package '{self.package_name}'. Returning fallback answer.")
            return "I could not find relevant code files for your request."
        documents = []
        text_splitter = CodeAwareTextSplitter(chunk_size=self.code_chunk_size, chunk_overlap=self.code_chunk_overlap)
        for file_name in relevant_files:
            file_path = self._find_file_path(file_name)
            if file_path and os.path.exists(file_path):
                logger.info(f"Found annotated script file: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        logger.warning(f"Annotated script file '{file_path}' is empty.")
                        continue
                    docs = text_splitter.split_text(content)
                    logger.info(f"Split '{file_path}' into {len(docs)} chunks.")
                    for i, doc_chunk in enumerate(docs):
                        documents.append(Document(page_content=doc_chunk, metadata={"source_file": file_name, "chunk_id": i}))
            else:
                logger.warning(f"Annotated script file '{file_name}' does not exist in any subdirectories.")
        if not documents:
            logger.info(f"No documents found to answer the query in second stage for package '{self.package_name}'.")
            return "I could not find relevant code content."
        logger.info(f"Indexed {len(documents)} documents for retrieval.")
        embeddings = GPT4AllEmbeddings(model=self.query_processing_model)
        collection_name = f"{self.package_name}_annotated_docs_{int(time.time())}"
        try:
            vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, collection_name=collection_name, client=self.chroma_client)
            logger.info(f"Documents indexed into temporary collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to index documents into ChromaDB: {str(e)}", exc_info=True)
            return "An error occurred while processing your request."
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.top_k_chunks})
        logger.info(f"Retriever created with top_k_chunks={self.top_k_chunks}.")
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "{context}\n\n"
                "User Request: {question}\n\n"
                "You are an autonomous Python coding agent. Your response must include three sections:\n"
                "1. **Query Understanding**: Briefly summarize what the user's query is asking.\n"
                "2. **Code Snippet**: Provide a robust, optimized, production-ready Python code solution that adheres strictly to PEP8 standards.\n"
                "3. **Code Explanation**: Offer a concise explanation of the code, including key design decisions and any enhancements made.\n\n"
                "Return your answer as three code blocks with 'Task_understanding' written in markdown format, 'Code_snippet' written in python, and 'Code_explanation' written in markdown format."
            )
        )
        try:
            llm = get_llm(self.query_processing_model)
        except Exception as e:
            logger.error("Failed to initialize LLM", exc_info=True)
            return "An error occurred while initializing the language model."
        try:
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt_template}
            )
            logger.info("RetrievalQA chain created successfully.")
        except Exception as e:
            logger.error(f"Failed to create RetrievalQA chain: {str(e)}", exc_info=True)
            return "An error occurred while setting up the retrieval chain."
        try:
            logger.info(f"Running retrieval QA for query: '{query}' in package '{self.package_name}'.")
            answer = chain.run(query)
            logger.info("Retrieval QA chain executed successfully.")
        except Exception as e:
            logger.error(f"Error during retrieval QA execution: {str(e)}", exc_info=True)
            return "An error occurred while generating the answer."
        if not isinstance(answer, str):
            logger.warning(f"Expected answer to be a string, got {type(answer)}. Converting to string.")
            answer = str(answer) if answer is not None else ""
        global_token_counter.count_output_tokens(answer, user=self.package_name)
        try:
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Temporary collection '{collection_name}' deleted successfully.")
        except chromadb.errors.InvalidCollectionException as e:
            logger.warning(f"Collection '{collection_name}' not found during cleanup for package '{self.package_name}': {e}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary collection '{collection_name}' for package '{self.package_name}': {e}")
        logger.info("Second stage completed successfully.")
        return answer

class RAGSystem:
    """
    Main RAG system managing multiple packages and their respective RAG stages.
    """
    def __init__(
            self,
            package_configs: List[PackageConfig],
            persist_dir: str = "./chroma_db_code",
            top_k_files: int = 5,
            top_k_chunks: int = 3,
            code_chunk_size: int = 2000,
            code_chunk_overlap: int = 200
    ):
        logger.info("Initializing RAG System for multiple Python packages...")
        self.package_configs = package_configs
        self.persist_directory = persist_dir
        self.top_k_files = top_k_files
        self.top_k_chunks = top_k_chunks
        self.code_chunk_size = code_chunk_size
        self.code_chunk_overlap = code_chunk_overlap
        self._validate_package_configs(package_configs)
        os.makedirs(self.persist_directory, exist_ok=True)
        self.chroma_client = chromadb.Client(
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=self.persist_directory
            )
        )
        self.first_stages = {}
        self.second_stages = {}
        for config in package_configs:
            logger.info(f"Setting up RAG for package: {config.name}")
            self.first_stages[config.name] = FirstStageRAG(
                converted_jsons_directory=config.converted_jsons_directory,
                persist_dir=self.persist_directory,
                file_selection_model=config.file_selection_model,
                chroma_client=self.chroma_client,
                top_k_files=self.top_k_files,
                package_name=config.name
            )
            self.second_stages[config.name] = SecondStageRAG(
                annotated_scripts_directory=config.annotated_scripts_directory,
                query_processing_model=config.query_processing_model,
                chroma_client=self.chroma_client,
                top_k_chunks=self.top_k_chunks,
                code_chunk_size=self.code_chunk_size,
                code_chunk_overlap=self.code_chunk_overlap,
                package_name=config.name
            )
        logger.info("RAG System for multiple packages initialized successfully.")

    def _validate_package_configs(self, package_configs: List[PackageConfig]):
        for config in package_configs:
            for directory in [config.converted_jsons_directory, config.annotated_scripts_directory]:
                if not os.path.exists(directory):
                    raise ValueError(f"Directory does not exist for package '{config.name}': {directory}")
                if not os.path.isdir(directory):
                    raise ValueError(f"Path is not a directory for package '{config.name}': {directory}")
                if not os.access(directory, os.R_OK):
                    raise ValueError(f"Directory is not readable for package '{config.name}': {directory}")
        logger.info("All package directories validated successfully.")

    def find_relevant_files(self, package_name: str, query: Union[str, Dict[str, Any]]) -> List[str]:
        if package_name not in self.first_stages:
            logger.error(f"Package '{package_name}' is not supported.")
            return []
        return self.first_stages[package_name].find_relevant_files(query)

    def answer_query_with_annotated_scripts(self, package_name: str, query: str, relevant_files: List[str]) -> str:
        if package_name not in self.second_stages:
            logger.error(f"Package '{package_name}' is not supported.")
            return "Unsupported package selected."
        return self.second_stages[package_name].answer_query_with_annotated_scripts(query, relevant_files)

    def generate_and_refine_code_sample(self, plot_info: dict, user_query: str) -> str:
        """
        Given plot metadata (including 'selected_package') and a user query/context,
        this method:
          1. Selects the appropriate package.
          2. Uses the first stage to find relevant files.
          3. Calls the second stage to generate a baseline code snippet.
          4. Refines the baseline code using the Gemini Stage2 model.
        Returns the final refined code snippet.
        """
        selected_package = plot_info.get('selected_package', None)
        if not selected_package or selected_package not in self.second_stages:
            logger.warning("Invalid or missing 'selected_package' in plot_info. Falling back to a default package.")
            selected_package = list(self.second_stages.keys())[0]
        relevant_files = self.find_relevant_files(selected_package, user_query)
        logger.info(f"Relevant files for package '{selected_package}': {relevant_files}")
        baseline_code = self.second_stages[selected_package].answer_query_with_annotated_scripts(user_query, relevant_files)
        logger.info("Baseline code sample generated.")
        refined_code = refine_code_with_gemini(baseline_code, user_query)
        logger.info("Baseline code sample refined using Gemini.")
        return refined_code

    def cleanup(self):
        logger.info("Cleaning up resources for all packages...")
        try:
            self.chroma_client.reset()
            logger.info("ChromaDB client reset successfully.")
            if os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory, ignore_errors=True)
                logger.info(f"Persistence directory '{self.persist_directory}' deleted successfully.")
            global_token_counter.save_counters()
            logger.info("Cleanup completed.")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
            raise

# --- END OF FILE rag_system.py ---