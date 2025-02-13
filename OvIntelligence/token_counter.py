# --- START OF FILE token_counter.py ---
import tiktoken
import logging
import json  # Added to enable JSON conversion
from typing import Tuple

# Set up logging for the token counter
logger = logging.getLogger('token_counter')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/token_counter.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


class TokenCounter:
    """
    A class to count tokens for input and output texts using tiktoken.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initializes the TokenCounter with the appropriate encoding based on the model.

        Args:
            model_name (str): The name of the model to determine encoding. Defaults to "gpt-3.5-turbo".
        """
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
            logger.info(f"Initialized TokenCounter with encoding for model '{model_name}'.")
        except KeyError:
            # Fallback to a default encoding if the model is not found
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Model '{model_name}' not found. Using default 'cl100k_base' encoding.")

        # Initialize counters
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text.

        Args:
            text (str): The text to count tokens for.

        Returns:
            int: The number of tokens.
        """
        if not isinstance(text, str):
            logger.error(f"Expected text to be a string, got {type(text)} instead.")
            return 0
        if not text.strip():
            return 0
        try:
            token_count = len(self.encoding.encode(text))
            logger.debug(f"Counted {token_count} tokens for text: {text[:50]}{'...' if len(text) > 50 else ''}")
            return token_count
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}", exc_info=True)
            return 0

    def count_input_tokens(self, input_text: str, user: str = "unknown") -> int:
        """
        Counts and updates the total input tokens.

        Args:
            input_text (str): The input text (e.g., user query).
            user (str): The identifier for the user or package.

        Returns:
            int: The number of tokens in the input text.
        """
        tokens = self.count_tokens(input_text)
        self.total_input_tokens += tokens
        logger.info(f"User '{user}' input tokens: {tokens}, Total input tokens: {self.total_input_tokens}")
        return tokens

    def count_output_tokens(self, output_text: str, user: str = "unknown") -> int:
        """
        Counts and updates the total output tokens.

        Args:
            output_text (str): The output text (e.g., model response).
            user (str): The identifier for the user or package.

        Returns:
            int: The number of tokens in the output text.
        """
        tokens = self.count_tokens(output_text)
        self.total_output_tokens += tokens
        logger.info(f"User '{user}' output tokens: {tokens}, Total output tokens: {self.total_output_tokens}")
        return tokens

    def get_total_tokens(self) -> Tuple[int, int]:
        """
        Retrieves the total input and output tokens counted so far.

        Returns:
            Tuple[int, int]: Total input tokens and total output tokens.
        """
        return self.total_input_tokens, self.total_output_tokens

    def reset_counters(self):
        """
        Resets the input and output token counters.
        """
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        logger.info("Token counters have been reset.")


# Singleton instance for global usage
global_token_counter = TokenCounter()


def count_tokens_decorator(func):
    """
    A decorator to automatically count input and output tokens for functions that process queries.

    Assumes that the decorated function returns the output text as a string.
    If not, the output is converted to a string using JSON serialization (if possible) or str().

    Args:
        func (callable): The function to decorate.

    Returns:
        callable: The wrapped function.
    """

    def wrapper(*args, **kwargs):
        # Extract the current user from keyword arguments or default to "unknown"
        user = kwargs.get('user', "unknown")
        input_text = kwargs.get('query') or (args[0] if args else "")
        input_tokens = global_token_counter.count_input_tokens(input_text, user)

        output_text = func(*args, **kwargs)

        # Convert non-string outputs to a string for token counting
        if not isinstance(output_text, str):
            try:
                output_text_str = json.dumps(output_text)
            except Exception:
                output_text_str = str(output_text)
        else:
            output_text_str = output_text

        output_tokens = global_token_counter.count_output_tokens(output_text_str, user)
        logger.info(
            f"Function '{func.__name__}' called by '{user}' with input tokens: {input_tokens}, output tokens: {output_tokens}")
        return output_text

    return wrapper

# --- END OF FILE token_counter.py ---