# Configuration Guide

Configure BioinformaticsDataCollector for optimal performance and reliability.

## Configuration Methods

Configuration options are applied in this priority order:

1. **Command-line arguments** (highest priority)
2. **Environment variables** 
3. **`.env` file**
4. **Default values** (lowest priority)

## Quick Setup

### 1. Basic .env File
Create a `.env` file in your project root:

```env
# Database (required)
DATABASE_URL=sqlite:///./biocollect.db

# Logging (recommended)
LOG_LEVEL=INFO
LOG_FILE=./biocollect.log

# Storage directories (optional)
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed
```

### 2. With API Keys (Recommended)
API keys provide better rate limits and access to additional features:

```env
# API Keys - Get these from respective services
UNIPROT_API_KEY=your_uniprot_key
NCBI_API_KEY=your_ncbi_api_key

# Database
DATABASE_URL=sqlite:///./biocollect.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=./biocollect.log

# Storage
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed
```

## Database Configuration

### SQLite (Default)
Best for single-user, local usage:
```env
DATABASE_URL=sqlite:///./biocollect.db
```

### PostgreSQL (Production)
Best for multi-user or server deployments:
```env
DATABASE_URL=postgresql://username:password@localhost:5432/biocollect
```

Example PostgreSQL setup:
```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb biocollect
sudo -u postgres createuser -P your_username

# Grant permissions
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE biocollect TO your_username;"
```

## API Keys & Rate Limits

### Getting API Keys

Most APIs work without keys but provide better performance with them:

#### NCBI E-utilities
- **Get key**: https://www.ncbi.nlm.nih.gov/account/
- **Benefit**: 10 requests/second vs 3 without key
- **Usage**: Ensembl, ClinVar, dbSNP, PubMed data

#### UniProt
- **Get key**: Contact UniProt support for high-volume usage
- **Benefit**: Higher rate limits for batch processing
- **Usage**: Protein sequences, annotations, GO terms

#### EBI Services
- **Get key**: https://www.ebi.ac.uk/Tools/webservices/
- **Benefit**: Priority access and higher limits
- **Usage**: InterPro, EMDB, PRIDE, Reactome

### Rate Limit Configuration
```env
# Default rates (requests per second)
UNIPROT_RATE_LIMIT=10
PDB_RATE_LIMIT=10
NCBI_RATE_LIMIT=3     # 10 with API key
ENSEMBL_RATE_LIMIT=15
KEGG_RATE_LIMIT=1     # KEGG is strict
```

## Logging Configuration

### Basic Logging
```env
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
LOG_FILE=./biocollect.log        # File path for logs
```

### Advanced Logging
```env
LOG_LEVEL=INFO
LOG_FILE=./logs/biocollect.log
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_MAX_SIZE=10MB                # Rotate log files
LOG_BACKUP_COUNT=5               # Keep 5 backup files
```

For debugging API issues:
```env
LOG_LEVEL=DEBUG
```

## Storage Configuration

### Directory Structure
```env
RAW_DATA_DIR=./data/raw          # Downloaded files (PDB, sequences)
PROCESSED_DATA_DIR=./data/processed  # Processed/transformed data
CACHE_DIR=./cache                # API response cache
```

### Custom Storage Locations
```env
# Store on external drive
RAW_DATA_DIR=/mnt/external/biocollect/raw
PROCESSED_DATA_DIR=/mnt/external/biocollect/processed

# Network storage
RAW_DATA_DIR=/nfs/shared/biocollect/raw
```

## Performance Tuning

### Network & Timeout Settings
```env
API_TIMEOUT=30                   # Request timeout (seconds)
API_MAX_RETRIES=3               # Retry failed requests
API_RETRY_DELAY=1               # Delay between retries (seconds)
```

### Batch Processing
```env
BATCH_SIZE=100                  # Process items in batches
MAX_CONCURRENT_REQUESTS=5       # Parallel API requests
```

### Database Performance
```env
# For PostgreSQL
DATABASE_POOL_SIZE=10           # Connection pool size
DATABASE_MAX_OVERFLOW=20        # Extra connections when needed

# For SQLite
DATABASE_TIMEOUT=30             # Lock timeout (seconds)
```

## Environment-Specific Configurations

### Development Environment
```env
DATABASE_URL=sqlite:///./dev_biocollect.db
LOG_LEVEL=DEBUG
LOG_FILE=./dev_biocollect.log
UNIPROT_RATE_LIMIT=2            # Be gentle during testing
PDB_RATE_LIMIT=2
```

### Production Environment
```env
DATABASE_URL=postgresql://user:pass@db-server:5432/biocollect
LOG_LEVEL=WARNING
LOG_FILE=/var/log/biocollect/production.log
UNIPROT_RATE_LIMIT=10
PDB_RATE_LIMIT=10
RAW_DATA_DIR=/data/biocollect/raw
PROCESSED_DATA_DIR=/data/biocollect/processed
```

### Testing Environment
```env
DATABASE_URL=sqlite:///:memory:  # In-memory database
LOG_LEVEL=ERROR                  # Minimal logging during tests
UNIPROT_RATE_LIMIT=1            # Slow down for testing
```

## Complete Configuration Example

### Production .env File
```env
# Database
DATABASE_URL=postgresql://biocollect:secure_password@localhost:5432/biocollect

# API Keys
UNIPROT_API_KEY=your_uniprot_key
NCBI_API_KEY=your_ncbi_key

# Storage
RAW_DATA_DIR=/data/biocollect/raw
PROCESSED_DATA_DIR=/data/biocollect/processed
CACHE_DIR=/data/biocollect/cache

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/biocollect/biocollect.log
LOG_MAX_SIZE=50MB
LOG_BACKUP_COUNT=10

# Performance
UNIPROT_RATE_LIMIT=10
NCBI_RATE_LIMIT=10
PDB_RATE_LIMIT=10
API_TIMEOUT=60
API_MAX_RETRIES=5
BATCH_SIZE=500
MAX_CONCURRENT_REQUESTS=10

# Database Performance
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
```

## Validation & Testing

### Test Your Configuration
```bash
# Test database connection
biocollect status

# Test API connections
biocollect collect uniprot P04637 --debug

# Validate all settings
biocollect config --validate
```

### Common Issues

#### Database Connection Failed
```bash
# Check DATABASE_URL format
# SQLite: sqlite:///./path/to/file.db
# PostgreSQL: postgresql://user:pass@host:port/dbname

# Verify file permissions for SQLite
chmod 664 biocollect.db
```

#### API Rate Limiting
```bash
# Reduce rate limits if getting 429 errors
export UNIPROT_RATE_LIMIT=5
export NCBI_RATE_LIMIT=2
```

#### Disk Space Issues
```bash
# Check available space
df -h

# Clean old cache files
rm -rf ./cache/*

# Move data directories to larger disk
export RAW_DATA_DIR=/mnt/storage/biocollect
```

## Security Best Practices

### API Keys
- Store API keys in `.env` file, never in code
- Use different keys for development and production  
- Rotate keys periodically
- Don't commit `.env` files to version control

### Database Security
- Use strong passwords for database connections
- Enable SSL for PostgreSQL connections
- Regularly backup database files
- Restrict database access to necessary users

### File Permissions
```bash
# Secure your .env file
chmod 600 .env

# Secure data directories
chmod 750 ./data
```