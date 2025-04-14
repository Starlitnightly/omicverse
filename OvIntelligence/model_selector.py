# --- START OF FILE model_selector.py ---
import os
import logging
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)

def get_llm(model_name: str):
    """
    Given a model name, instantiate and return the appropriate LLM instance.
    
    For Gemini models (model names starting with 'gemini'):
      - Configures the API key (from GEMINI_API_KEY env variable)
      - Returns a ChatGoogleGenerativeAI instance.
    
    For other models:
      - Returns an Ollama instance.
    
    Args:
        model_name (str): The name of the model to use.
    
    Returns:
        An instance of the selected language model.
    
    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set for Gemini models.
        Exception: If initialization of the model fails.
    """
    if model_name.lower().startswith("gemini"):
        try:
            api_key = os.environ["GEMINI_API_KEY"]
        except KeyError as e:
            logger.error("GEMINI_API_KEY environment variable is not set.")
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.") from e

        genai.configure(api_key=api_key)
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.7,
                google_api_key=api_key
            )
            logger.info(f"Gemini LLM initialized with model '{model_name}'.")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {str(e)}", exc_info=True)
            raise e
    else:
        try:
            llm = Ollama(
                model=model_name,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )
            logger.info(f"Ollama LLM initialized with model '{model_name}'.")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {str(e)}", exc_info=True)
            raise e

# --- END OF FILE model_selector.py ---