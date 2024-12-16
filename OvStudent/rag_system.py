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

# Use the original GPT4AllEmbeddings and Ollama LLM from the initial code.
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

def setup_logging():
    logger = logging.getLogger('rag_system_code_optimized')
    logger.setLevel(logging.INFO)

    os.makedirs('logs', exist_ok=True)

    file_handler = RotatingFileHandler(
        'logs/rag_system_code_optimized.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
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


class CodeAwareTextSplitter(CharacterTextSplitter):
    """
    A custom text splitter that tries to split code more gracefully.
    You can enhance this by:
    - Splitting on `def `, `class ` boundaries.
    - Avoiding splitting in the middle of a function.
    For now, this is a placeholder that uses line-based splitting
    but could be improved as needed.
    """
    def __init__(self, chunk_size=2000, chunk_overlap=200):
        super().__init__(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class FirstStageRAG:
    def __init__(
            self,
            converted_jsons_directory: str,
            persist_dir: str,
            file_selection_model: str,
            chroma_client: chromadb.Client,
            top_k_files: int = 3
    ):
        self.converted_jsons_directory = converted_jsons_directory
        self.persist_dir = persist_dir
        self.file_selection_model = file_selection_model
        self.chroma_client = chroma_client
        self.top_k_files = top_k_files

        logger.info("Initializing FirstStageRAG for code retrieval...")

        self.collection_name = "file_descriptions"
        self.collection = self._load_or_create_collection()

        if self.collection.count() == 0:
            logger.info("Populating first-stage collection with file descriptions...")
            self._index_file_descriptions()
            logger.info("File descriptions indexed successfully.")

        logger.info("FirstStageRAG initialized successfully.")

    def _load_or_create_collection(self):
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            return collection
        except:
            logger.info(f"Creating a new Chroma collection: {self.collection_name}")
            collection = self.chroma_client.create_collection(name=self.collection_name)
            return collection

    def _index_file_descriptions(self):
        embeddings = GPT4AllEmbeddings(model=self.file_selection_model)

        docs = []
        metadatas = []
        ids = []

        for fname in os.listdir(self.converted_jsons_directory):
            if fname.endswith(".json"):
                fpath = os.path.join(self.converted_jsons_directory, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        description = data.get("description", "")
                        file_name = data.get("file", fname)

                        if not description:
                            continue

                        docs.append(description)
                        metadatas.append({"file": file_name})
                        ids.append(file_name)
                    except Exception as e:
                        logger.error(f"Error reading {fpath}: {str(e)}", exc_info=True)

        if docs:
            doc_embeddings = embeddings.embed_documents(docs)
            self.collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=doc_embeddings)
        else:
            logger.warning("No documents to index in FirstStageRAG.")

    def find_relevant_files(self, query: Union[str, Dict[str, Any]]) -> List[str]:
        if isinstance(query, dict):
            query_text = query.get("query", "")
        else:
            query_text = query

        if not query_text.strip():
            logger.warning("Empty query received in FirstStageRAG.")
            return []

        embeddings = GPT4AllEmbeddings(model=self.file_selection_model)
        query_embedding = embeddings.embed_query(query_text)

        results = self.collection.query(query_embeddings=[query_embedding], n_results=self.top_k_files)

        matched_files = []
        if results and "metadatas" in results and results["metadatas"]:
            for meta in results["metadatas"][0]:
                file_name = meta.get("file", None)
                if file_name:
                    matched_files.append(file_name)

        logger.info(f"Top {self.top_k_files} matched files for query '{query_text}': {matched_files}")
        return matched_files


class SecondStageRAG:
    def __init__(
            self,
            annotated_scripts_directory: str,
            query_processing_model: str,
            chroma_client: chromadb.Client,
            code_chunk_size: int = 2000,
            code_chunk_overlap: int = 200,
            top_k_chunks: int = 3
    ):
        self.annotated_scripts_directory = annotated_scripts_directory
        self.query_processing_model = query_processing_model
        self.chroma_client = chroma_client
        self.code_chunk_size = code_chunk_size
        self.code_chunk_overlap = code_chunk_overlap
        self.top_k_chunks = top_k_chunks

        logger.info("Initializing SecondStageRAG for code generation...")

        logger.info("SecondStageRAG initialized successfully.")

    def answer_query_with_annotated_scripts(self, query: str, relevant_files: List[str]) -> str:
        if not relevant_files:
            logger.info("No relevant files provided to second stage. Returning fallback answer.")
            return "I could not find relevant code files for your request."

        documents = []
        text_splitter = CodeAwareTextSplitter(chunk_size=self.code_chunk_size, chunk_overlap=self.code_chunk_overlap)

        for file_name in relevant_files:
            file_path = os.path.join(self.annotated_scripts_directory, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                docs = text_splitter.split_text(content)
                for i, doc_chunk in enumerate(docs):
                    documents.append(
                        Document(
                            page_content=doc_chunk,
                            metadata={"source_file": file_name, "chunk_id": i}
                        )
                    )
            else:
                logger.warning(f"File {file_name} not found in annotated scripts directory.")

        if not documents:
            logger.info("No documents found to answer the query in second stage.")
            return "I could not find relevant code content."

        embeddings = GPT4AllEmbeddings(model=self.query_processing_model)

        collection_name = f"annotated_docs_{int(time.time())}"
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            client=self.chroma_client
        )

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.top_k_chunks})

        # System prompt and template for code generation
        system_prompt = (
            "You are a Python code generation assistant. "
            "You will be provided with context code snippets related to the user query. "
            "Please produce clear, correct, and well-commented Python code that addresses the user's request. "
            "Follow PEP8 standards. If you propose changes, ensure they run without syntax errors."
        )

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "{context}\n\n"
                "User Request: {question}\n\n"
                "Now generate the best possible Python code solution given the above context. "
                "If appropriate, include function definitions, classes, or usage examples. "
                "Make sure the final answer is strictly Python code."
            )
        )

        llm = Ollama(
            model=self.query_processing_model,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt_template}
        )

        logger.info(f"Running retrieval QA for code generation query: {query}")
        answer = chain.run(query)

        # Cleanup the temporary collection after use
        try:
            self.chroma_client.delete_collection(name=collection_name)
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary collection {collection_name}: {e}")

        return answer


class RAGSystem:
    def __init__(
            self,
            converted_jsons_directory: str,
            annotated_scripts_directory: str,
            persist_dir: str = "./chroma_db_code",
            file_selection_model: str = "qwen2.5-coder:3b",
            query_processing_model: str = "qwen2.5-coder:7b",
            top_k_files: int = 3,
            top_k_chunks: int = 3,
            code_chunk_size: int = 2000,
            code_chunk_overlap: int = 200
    ):
        logger.info("Initializing RAG System for Python code generation...")

        self.converted_jsons_directory = converted_jsons_directory
        self.annotated_scripts_directory = annotated_scripts_directory
        self.persist_directory = persist_dir
        self.file_selection_model = file_selection_model
        self.query_processing_model = query_processing_model
        self.top_k_files = top_k_files
        self.top_k_chunks = top_k_chunks
        self.code_chunk_size = code_chunk_size
        self.code_chunk_overlap = code_chunk_overlap

        self._validate_directories()

        os.makedirs(self.persist_directory, exist_ok=True)

        self.chroma_client = chromadb.Client(
            chromadb.config.Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=self.persist_directory
            )
        )

        self.first_stage = FirstStageRAG(
            converted_jsons_directory=self.converted_jsons_directory,
            persist_dir=self.persist_directory,
            file_selection_model=self.file_selection_model,
            chroma_client=self.chroma_client,
            top_k_files=self.top_k_files
        )

        self.second_stage = SecondStageRAG(
            annotated_scripts_directory=self.annotated_scripts_directory,
            query_processing_model=self.query_processing_model,
            chroma_client=self.chroma_client,
            top_k_chunks=self.top_k_chunks,
            code_chunk_size=self.code_chunk_size,
            code_chunk_overlap=self.code_chunk_overlap
        )

        logger.info("RAG System for code generation initialized successfully")

    def _validate_directories(self):
        for directory in [self.converted_jsons_directory, self.annotated_scripts_directory]:
            if not os.path.exists(directory):
                raise ValueError(f"Directory does not exist: {directory}")
            if not os.path.isdir(directory):
                raise ValueError(f"Path is not a directory: {directory}")
            if not os.access(directory, os.R_OK):
                raise ValueError(f"Directory is not readable: {directory}")
        logger.info("All directories validated successfully.")

    def find_relevant_files(self, query: Union[str, Dict[str, Any]]) -> List[str]:
        return self.first_stage.find_relevant_files(query)

    def answer_query_with_annotated_scripts(
            self,
            query: str,
            relevant_files: List[str]
    ) -> str:
        return self.second_stage.answer_query_with_annotated_scripts(query, relevant_files)

    def cleanup(self):
        logger.info("Cleaning up resources...")
        try:
            self.chroma_client.reset()
            if os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory, ignore_errors=True)
            logger.info("Cleanup completed.")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
            raise

# --- END OF FILE rag_system_code.py ---
