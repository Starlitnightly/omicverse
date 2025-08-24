"""Logging configuration and utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

from ..config.config import settings


# Install rich traceback handler
install_rich_traceback(show_locals=True)

# Create console for rich output
console = Console()


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        use_rich: Whether to use rich formatting
    """
    # Use settings if not provided
    log_level = log_level or settings.logging.log_level
    log_file = log_file or settings.logging.log_file
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create handlers list
    handlers = []
    
    # Console handler
    if use_rich:
        console_handler = RichHandler(
            console=console,
            show_time=settings.logging.show_time,
            show_path=settings.logging.show_path,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(settings.logging.log_format)
        )
    
    handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(settings.logging.log_format)
        )
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,
    )
    
    # Set levels for specific loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ProgressLogger:
    """Logger with progress tracking capabilities."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = get_logger(self.__class__.__name__)
        
        # Create progress bar if using rich
        try:
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
            
            self.progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
                console=console,
            )
            self.task = self.progress.add_task(description, total=total)
            self.use_rich = True
        except ImportError:
            self.progress = None
            self.use_rich = False
    
    def __enter__(self):
        if self.use_rich:
            self.progress.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_rich:
            self.progress.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, advance: int = 1, message: Optional[str] = None):
        """Update progress.
        
        Args:
            advance: Number of items completed
            message: Optional status message
        """
        self.current += advance
        
        if self.use_rich:
            self.progress.update(self.task, advance=advance)
        else:
            percentage = (self.current / self.total) * 100
            self.logger.info(
                f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)"
            )
        
        if message:
            self.logger.info(message)
    
    def complete(self, message: Optional[str] = None):
        """Mark progress as complete."""
        remaining = self.total - self.current
        if remaining > 0:
            self.update(remaining)
        
        if message:
            self.logger.info(message)


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log an error with context.
    
    Args:
        logger: Logger instance
        error: Exception to log
        context: Additional context
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    if context:
        logger.error(f"{context}: {error_type}: {error_msg}")
    else:
        logger.error(f"{error_type}: {error_msg}")
    
    # Log traceback at debug level
    logger.debug("Traceback:", exc_info=True)


def log_api_call(logger: logging.Logger, method: str, url: str, status: int = None):
    """Log API call details.
    
    Args:
        logger: Logger instance
        method: HTTP method
        url: Request URL
        status: Response status code
    """
    if status:
        logger.debug(f"{method} {url} -> {status}")
    else:
        logger.debug(f"{method} {url}")
