# Configuration Guide

This guide covers all configuration options for BioinformaticsDataCollector.

## Table of Contents

1. [Configuration Methods](#configuration-methods)
2. [Environment Variables](#environment-variables)
3. [Configuration File](#configuration-file)
4. [API Configuration](#api-configuration)
5. [Database Configuration](#database-configuration)
6. [Logging Configuration](#logging-configuration)
7. [Storage Configuration](#storage-configuration)
8. [Advanced Configuration](#advanced-configuration)

## Configuration Methods

BioinformaticsDataCollector supports multiple configuration methods, in order of precedence:

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **`.env` file**
4. **Configuration file** (`config.yaml`)
5. **Default values** (lowest priority)

### Example

```bash
# Command-line argument (highest priority)
biocollect collect uniprot P04637 --rate-limit 5

# Environment variable
export UNIPROT_RATE_LIMIT=5
biocollect collect uniprot P04637

# .env file
echo "UNIPROT_RATE_LIMIT=5" >> .env
biocollect collect uniprot P04637

# config.yaml (lowest priority)
api:
  uniprot:
    rate_limit: 5
```

## Environment Variables

### Basic Configuration

Create a `.env` file in your project root:

```bash
# API Keys
UNIPROT_API_KEY=your_uniprot_key
NCBI_API_KEY=your_ncbi_api_key
ENSEMBL_API_KEY=your_ensembl_key

# Database
DATABASE_URL=sqlite:///./biocollect.db
# For PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost/biocollect

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/biocollect.log
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Storage
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed
CACHE_DIR=./cache

# API Rate Limits
UNIPROT_RATE_LIMIT=10
PDB_RATE_LIMIT=10
NCBI_RATE_LIMIT=3

# Timeouts
API_TIMEOUT=30
API_MAX_RETRIES=3
```

### All Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///./biocollect.db` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `LOG_FILE` | Log file path | `None` (console only) |
| `RAW_DATA_DIR` | Directory for raw downloaded files | `./data/raw` |
| `PROCESSED_DATA_DIR` | Directory for processed data | `./data/processed` |
| `CACHE_DIR` | Directory for cached API responses | `./cache` |
| `UNIPROT_API_KEY` | UniProt API key (optional) | `None` |
| `UNIPROT_RATE_LIMIT` | UniProt requests per second | `10` |
| `PDB_RATE_LIMIT` | PDB requests per second | `10` |
| `NCBI_API_KEY` | NCBI E-utilities API key | `None` |
| `NCBI_RATE_LIMIT` | NCBI requests per second | `3` (10 with key) |
| `API_TIMEOUT` | API request timeout (seconds) | `30` |
| `API_MAX_RETRIES` | Maximum retry attempts | `3` |

## Configuration File

For complex configurations, use `config/config.yaml`:

```yaml
# API Configuration
api:
  uniprot:
    base_url: https://rest.uniprot.org
    rate_limit: 10
    timeout: 30
    max_retries: 3
    headers:
      User-Agent: BioinformaticsCollector/1.0
  
  pdb:
    base_url: https://data.rcsb.org
    search_url: https://search.rcsb.org
    download_url: https://files.rcsb.org
    rate_limit: 10
    timeout: 60
  
  ncbi:
    base_url: https://eutils.ncbi.nlm.nih.gov/entrez/eutils
    api_key: ${NCBI_API_KEY}  # Reference environment variable
    rate_limit: 10  # With API key
    timeout: 30

# Database Configuration
database:
  url: ${DATABASE_URL:sqlite:///./biocollect.db}
  echo: false  # SQL logging
  pool_size: 5
  max_overflow: 10
  pool_timeout: 30
  pool_recycle: 3600

# Storage Configuration
storage:
  raw_data_dir: ./data/raw
  processed_data_dir: ./data/processed
  cache_dir: ./cache
  structure_formats:
    - pdb
    - cif
    - mmtf
  compression: gzip

# Logging Configuration
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: ./logs/biocollect.log
  max_bytes: 10485760  # 10MB
  backup_count: 5
  rich_console: true  # Pretty console output
  
  # Module-specific levels
  loggers:
    src.api: DEBUG
    src.collectors: INFO
    sqlalchemy.engine: WARNING

# Cache Configuration
cache:
  enabled: true
  ttl: 3600  # 1 hour
  max_size: 1000  # Maximum cached items
  backends:
    - memory
    - disk

# Feature Flags
features:
  async_downloads: true
  batch_optimization: true
  auto_retry: true
  progress_bars: true
```

## API Configuration

### UniProt Configuration

```python
# config/config.py or environment
UNIPROT_BASE_URL = "https://rest.uniprot.org"
UNIPROT_RATE_LIMIT = 10  # requests per second
UNIPROT_TIMEOUT = 30  # seconds
UNIPROT_MAX_RETRIES = 3
UNIPROT_BATCH_SIZE = 100  # For batch operations

# Optional API key for higher limits
UNIPROT_API_KEY = "your_key_here"
```

### PDB Configuration

```python
PDB_BASE_URL = "https://data.rcsb.org"
PDB_SEARCH_URL = "https://search.rcsb.org"
PDB_DOWNLOAD_URL = "https://files.rcsb.org"
PDB_RATE_LIMIT = 10
PDB_TIMEOUT = 60  # Larger files need more time
PDB_STRUCTURE_FORMATS = ["pdb", "cif", "mmtf"]
```

### NCBI Configuration

```python
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = "your_ncbi_key"  # Required for > 3 req/sec
NCBI_RATE_LIMIT = 3  # Without key
# NCBI_RATE_LIMIT = 10  # With key
NCBI_TIMEOUT = 30
NCBI_RETRY_CODES = [429, 500, 502, 503, 504]
```

## Database Configuration

### SQLite (Default)

```python
DATABASE_URL = "sqlite:///./biocollect.db"

# SQLite-specific options
SQLITE_PRAGMA = {
    "journal_mode": "WAL",
    "cache_size": -64000,  # 64MB
    "foreign_keys": 1,
    "synchronous": "NORMAL"
}
```

### PostgreSQL

```python
DATABASE_URL = "postgresql://user:password@localhost:5432/biocollect"

# PostgreSQL-specific
POSTGRES_POOL_SIZE = 20
POSTGRES_MAX_OVERFLOW = 40
POSTGRES_POOL_TIMEOUT = 30
POSTGRES_POOL_RECYCLE = 3600
```

### MySQL/MariaDB

```python
DATABASE_URL = "mysql+pymysql://user:password@localhost:3306/biocollect"

# MySQL-specific
MYSQL_CHARSET = "utf8mb4"
MYSQL_POOL_SIZE = 10
```

### Database Connection Pooling

```python
# config/database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,  # Verify connections
    echo=False  # Set True for SQL logging
)
```

## Logging Configuration

### Basic Setup

```python
# Simple configuration
LOG_LEVEL = "INFO"
LOG_FILE = "./logs/biocollect.log"
```

### Advanced Setup

```python
# config/logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "biocollect.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "errors.log",
            "maxBytes": 10485760,
            "backupCount": 5
        }
    },
    "loggers": {
        "src": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False
        },
        "sqlalchemy.engine": {
            "level": "WARNING",
            "handlers": ["file"],
            "propagate": False
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file", "error_file"]
    }
}
```

### Using Rich for Pretty Output

```python
# Enable rich console output
RICH_CONSOLE = True
RICH_TRACEBACK = True
RICH_MARKUP = True

# Custom themes
RICH_THEME = {
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green"
}
```

## Storage Configuration

### Directory Structure

```python
# Base directories
STORAGE_BASE_DIR = "./data"
RAW_DATA_DIR = f"{STORAGE_BASE_DIR}/raw"
PROCESSED_DATA_DIR = f"{STORAGE_BASE_DIR}/processed"
CACHE_DIR = f"{STORAGE_BASE_DIR}/cache"

# Subdirectories
STRUCTURE_DIR = f"{RAW_DATA_DIR}/structures"
SEQUENCE_DIR = f"{RAW_DATA_DIR}/sequences"
ANNOTATION_DIR = f"{RAW_DATA_DIR}/annotations"

# File naming patterns
FILE_NAMING = {
    "protein": "{accession}_{version}.json",
    "structure": "{pdb_id}_{chain}.{format}",
    "sequence": "{accession}.fasta"
}
```

### Compression Settings

```python
# Enable compression for stored files
ENABLE_COMPRESSION = True
COMPRESSION_TYPE = "gzip"  # or "bz2", "lzma"
COMPRESSION_LEVEL = 6  # 1-9 (speed vs size)

# Compress files larger than threshold
COMPRESSION_THRESHOLD = 1024 * 1024  # 1MB
```

### Cache Configuration

```python
# Cache settings
CACHE_ENABLED = True
CACHE_BACKEND = "disk"  # or "memory", "redis"
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 1000  # entries

# Redis cache (if using)
REDIS_URL = "redis://localhost:6379/0"
REDIS_KEY_PREFIX = "biocollect:"
```

## Advanced Configuration

### Performance Tuning

```python
# Batch processing
BATCH_SIZE = 100
BATCH_TIMEOUT = 300  # 5 minutes
BATCH_RETRY_ON_PARTIAL = True

# Concurrent operations
MAX_WORKERS = 4
THREAD_POOL_SIZE = 10
ASYNC_ENABLED = True

# Memory management
MAX_MEMORY_MB = 1024  # Limit memory usage
CHUNK_SIZE = 1000  # Process in chunks
```

### Retry Configuration

```python
# Retry settings
RETRY_ENABLED = True
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff
RETRY_ON_CODES = [429, 500, 502, 503, 504]
RETRY_ON_EXCEPTIONS = [
    "ConnectionError",
    "Timeout",
    "ReadTimeout"
]
```

### Security Configuration

```python
# API key encryption
ENCRYPT_API_KEYS = True
ENCRYPTION_KEY = "your-secret-key"  # Use KMS in production

# SSL/TLS
VERIFY_SSL = True
SSL_CERT_PATH = "/path/to/cert.pem"

# Request headers
DEFAULT_HEADERS = {
    "User-Agent": "BioinformaticsCollector/1.0 (https://github.com/user/repo)",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate"
}
```

### Proxy Configuration

```python
# HTTP proxy
HTTP_PROXY = "http://proxy.example.com:8080"
HTTPS_PROXY = "http://proxy.example.com:8080"
NO_PROXY = "localhost,127.0.0.1,.internal.domain"

# SOCKS proxy
SOCKS_PROXY = "socks5://proxy.example.com:1080"
```

## Configuration in Code

### Using Pydantic Settings

```python
# config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API settings
    uniprot_base_url: str = "https://rest.uniprot.org"
    uniprot_api_key: Optional[str] = None
    uniprot_rate_limit: int = 10
    
    # Database
    database_url: str = "sqlite:///./biocollect.db"
    database_echo: bool = False
    
    # Storage
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Singleton instance
settings = Settings()
```

### Dynamic Configuration

```python
# config/dynamic.py
import os
from pathlib import Path

class DynamicConfig:
    """Configuration that can be updated at runtime."""
    
    def __init__(self):
        self._config = {}
        self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration."""
        self._config = {
            "rate_limits": {
                "uniprot": 10,
                "pdb": 10,
                "ncbi": 3
            },
            "timeouts": {
                "default": 30,
                "download": 300
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value):
        """Set configuration value."""
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_from_file(self, filepath: str):
        """Update configuration from YAML file."""
        import yaml
        
        with open(filepath) as f:
            data = yaml.safe_load(f)
        
        self._merge(self._config, data)
    
    def _merge(self, base: dict, update: dict):
        """Recursively merge dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge(base[key], value)
            else:
                base[key] = value

# Global dynamic config
dynamic_config = DynamicConfig()
```

## Environment-Specific Configuration

### Development

```bash
# .env.development
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///./dev.db
API_TIMEOUT=60
CACHE_ENABLED=False
```

### Production

```bash
# .env.production
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://user:pass@db.server/biocollect
API_TIMEOUT=30
CACHE_ENABLED=True
CACHE_BACKEND=redis
REDIS_URL=redis://redis.server:6379/0
```

### Testing

```bash
# .env.test
LOG_LEVEL=ERROR
DATABASE_URL=sqlite:///:memory:
API_TIMEOUT=5
MOCK_API_CALLS=True
```

## Configuration Validation

```python
# config/validate.py
def validate_config():
    """Validate configuration on startup."""
    errors = []
    
    # Check required directories exist
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR]:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True)
            except Exception as e:
                errors.append(f"Cannot create directory {dir_path}: {e}")
    
    # Validate database connection
    try:
        from sqlalchemy import create_engine
        engine = create_engine(DATABASE_URL)
        engine.connect()
    except Exception as e:
        errors.append(f"Database connection failed: {e}")
    
    # Check API keys if required
    if NCBI_RATE_LIMIT > 3 and not NCBI_API_KEY:
        errors.append("NCBI API key required for rate limit > 3")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    return True
```

## Troubleshooting Configuration

### Common Issues

1. **Missing Environment Variables**
   ```bash
   # Check loaded variables
   python -c "from config import settings; print(settings.dict())"
   ```

2. **Permission Errors**
   ```bash
   # Fix directory permissions
   chmod -R 755 ./data
   ```

3. **Database Connection Issues**
   ```python
   # Test connection
   from config import settings
   from sqlalchemy import create_engine
   
   engine = create_engine(settings.database_url, echo=True)
   engine.connect()
   ```

4. **API Rate Limiting**
   ```python
   # Check current limits
   from src.api.uniprot import UniProtClient
   
   client = UniProtClient()
   print(f"Rate limit: {client.rate_limit} req/sec")
   ```

### Debug Mode

Enable debug mode for detailed information:

```bash
# Environment variable
export DEBUG=True
export LOG_LEVEL=DEBUG

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```