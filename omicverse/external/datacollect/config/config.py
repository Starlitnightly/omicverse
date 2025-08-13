"""Configuration management for the bioinformatics data collector."""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class APIConfig(BaseSettings):
    """API configuration settings."""
    
    # UniProt
    uniprot_base_url: str = "https://rest.uniprot.org"
    
    # PDB
    pdb_base_url: str = "https://data.rcsb.org"
    pdb_search_url: str = "https://search.rcsb.org"
    
    # AlphaFold
    alphafold_base_url: str = "https://alphafold.ebi.ac.uk"
    
    # NCBI
    ncbi_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ncbi_api_key: Optional[str] = Field(None, env="NCBI_API_KEY")
    
    # AlphaFold
    alphafold_base_url: str = "https://alphafold.ebi.ac.uk/api"
    alphafold_rate_limit: int = 10
    
    # Ensembl
    ensembl_base_url: str = "https://rest.ensembl.org"
    
    # KEGG
    kegg_base_url: str = "https://rest.kegg.jp"
    
    # Rate limiting
    default_rate_limit: int = 10  # requests per second
    default_timeout: int = 30  # seconds
    max_retries: int = 3
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


class StorageConfig(BaseSettings):
    """Storage configuration settings."""
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    cache_dir: Path = data_dir / "cache"
    
    # Database
    database_url: str = Field(
        default="sqlite:///data/biocollect.db",
        env="DATABASE_URL"
    )
    
    # Cache settings
    cache_ttl: int = 86400  # 24 hours in seconds
    max_cache_size_mb: int = 1000
    
    @field_validator("data_dir", "raw_data_dir", "processed_data_dir", "cache_dir")
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class LoggingConfig(BaseSettings):
    """Logging configuration settings."""
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    
    # Rich console settings
    show_time: bool = True
    show_path: bool = True


class Settings:
    """Main settings class combining all configurations."""
    
    def __init__(self):
        self.api = APIConfig()
        self.storage = StorageConfig()
        self.logging = LoggingConfig()
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service."""
        return os.getenv(f"{service.upper()}_API_KEY")


# Global settings instance
settings = Settings()