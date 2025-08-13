"""Base collector with common functionality."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import hashlib
import json

from sqlalchemy.orm import Session

from omicverse.external.datacollect.models.base import get_db
from config import settings


logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Base class for all data collectors."""
    
    def __init__(self, api_client, db_session: Optional[Session] = None):
        self.api_client = api_client
        self._db_session = db_session
    
    @property
    def db_session(self) -> Session:
        """Get database session."""
        if self._db_session is None:
            self._db_session = next(get_db())
        return self._db_session
    
    def generate_id(self, *args) -> str:
        """Generate unique ID based on input arguments."""
        content = "_".join(str(arg) for arg in args)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @abstractmethod
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a single identifier."""
        pass
    
    @abstractmethod
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        pass
    
    @abstractmethod
    def save_to_database(self, data: Dict[str, Any]) -> Any:
        """Save collected data to database."""
        pass
    
    def save_to_file(self, data: Any, filename: str, format: str = "json"):
        """Save data to file.
        
        Args:
            data: Data to save
            filename: Output filename
            format: File format (json, csv, etc.)
        """
        output_path = settings.storage.processed_data_dir / filename
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "csv":
            import pandas as pd
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved data to {output_path}")
        return output_path
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate collected data.
        
        Override in subclasses for specific validation.
        """
        return True
    
    def process_and_save(self, identifier: str, **kwargs):
        """Collect, validate, and save data."""
        try:
            # Extract save_to_file parameter
            save_to_file = kwargs.pop("save_to_file", False)
            
            # Collect data with remaining kwargs
            data = self.collect_single(identifier, **kwargs)
            
            # Validate
            if not self.validate_data(data):
                logger.error(f"Data validation failed for {identifier}")
                return None
            
            # Save to database
            model_instance = self.save_to_database(data)
            
            # Optionally save to file
            if save_to_file:
                filename = f"{self.__class__.__name__}_{identifier}.json"
                self.save_to_file(data, filename)
            
            logger.info(f"Successfully processed {identifier}")
            return model_instance
            
        except Exception as e:
            logger.error(f"Error processing {identifier}: {e}")
            raise
    
    def close(self):
        """Clean up resources."""
        if self._db_session:
            self._db_session.close()
        if hasattr(self.api_client, "close"):
            self.api_client.close()