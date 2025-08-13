# Troubleshooting

Common issues and solutions for BioinformaticsDataCollector.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Database Problems](#database-problems)
3. [API and Network Issues](#api-and-network-issues)
4. [Data Collection Errors](#data-collection-errors)
5. [Performance Issues](#performance-issues)
6. [Platform-Specific Issues](#platform-specific-issues)
7. [Getting Help](#getting-help)

## Installation Issues

### Python Version Compatibility

#### Issue: "Python version not supported"
```bash
ERROR: This package requires Python >= 3.9
```

**Solution:**
```bash
# Check your Python version
python --version

# Install Python 3.9+ if needed
# Using pyenv (recommended)
pyenv install 3.9.18
pyenv local 3.9.18

# Or using conda
conda create -n biocollect python=3.9
conda activate biocollect
```

### Package Installation Failures

#### Issue: "Failed building wheel for package"
```bash
ERROR: Failed building wheel for psycopg2
```

**Solutions:**
```bash
# For PostgreSQL dependencies (Linux/MacOS)
# Ubuntu/Debian
sudo apt-get install libpq-dev python3-dev

# CentOS/RHEL
sudo yum install postgresql-devel python3-devel

# MacOS
brew install postgresql

# Then retry installation
pip install -r requirements.txt
```

#### Issue: "Microsoft Visual C++ 14.0 is required" (Windows)
**Solution:**
1. Install Microsoft C++ Build Tools
2. Or use conda instead of pip:
```bash
conda install -c conda-forge datacollect2bionmi
```

### Virtual Environment Issues

#### Issue: "Command 'biocollect' not found"
**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/MacOS
# or
venv\Scripts\activate  # Windows

# Reinstall in development mode
pip install -e .

# Verify installation
which biocollect
biocollect --help
```

## Database Problems

### Database Initialization

#### Issue: "Database connection failed. Run 'biocollect init' first."
**Solution:**
```bash
# Initialize the database
biocollect init

# If that fails, check database URL
echo $DATABASE_URL

# For SQLite, ensure directory exists
mkdir -p $(dirname ./biocollect.db)
biocollect init
```

#### Issue: "Permission denied" for SQLite database
**Solution:**
```bash
# Check file permissions
ls -la biocollect.db

# Fix permissions
chmod 664 biocollect.db
chown $USER:$USER biocollect.db

# Or recreate database
rm biocollect.db
biocollect init
```

### PostgreSQL Connection Issues

#### Issue: "FATAL: password authentication failed"
**Solution:**
```bash
# Check connection string format
DATABASE_URL=postgresql://username:password@localhost:5432/dbname

# Test connection manually
psql -h localhost -U username -d dbname

# Reset password if needed
sudo -u postgres psql
\password username
```

#### Issue: "could not connect to server"
**Solution:**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql  # Linux
brew services list | grep postgres  # MacOS

# Start PostgreSQL if stopped
sudo systemctl start postgresql  # Linux
brew services start postgresql  # MacOS

# Check connection
telnet localhost 5432
```

### Database Lock Issues (SQLite)

#### Issue: "database is locked"
```python
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) database is locked
```

**Solution:**
```bash
# Find processes using the database
lsof biocollect.db

# Kill hanging processes if safe
kill -9 <PID>

# Or use a different database file
export DATABASE_URL=sqlite:///./biocollect_new.db
biocollect init
```

### Migration and Schema Issues

#### Issue: "table doesn't exist" after update
**Solution:**
```python
# Recreate database schema
from src.utils.database import initialize_database
initialize_database()

# Or migrate manually
python -c "
from src.models.base import Base, engine
Base.metadata.create_all(engine)
"
```

## API and Network Issues

### Rate Limiting

#### Issue: "Too Many Requests (429)"
```bash
HTTPError: 429 Client Error: Too Many Requests
```

**Solution:**
```bash
# Reduce rate limits in .env file
echo "UNIPROT_RATE_LIMIT=2" >> .env
echo "NCBI_RATE_LIMIT=1" >> .env

# Or use API keys for higher limits
echo "UNIPROT_API_KEY=your_key" >> .env
echo "NCBI_API_KEY=your_key" >> .env
```

### Network Connectivity

#### Issue: "Connection timeout" or "Network unreachable"
**Solution:**
```bash
# Test basic connectivity
ping rest.uniprot.org
ping eutils.ncbi.nlm.nih.gov

# Check proxy settings
echo $HTTP_PROXY
echo $HTTPS_PROXY

# Configure proxy if needed
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=https://proxy.company.com:8080
```

### SSL Certificate Issues

#### Issue: "SSL: CERTIFICATE_VERIFY_FAILED"
**Solution:**
```bash
# Update certificates (MacOS)
/Applications/Python\ 3.9/Install\ Certificates.command

# For corporate networks, add certificate
export SSL_CERT_FILE=/path/to/corporate-ca-bundle.crt

# Or temporarily disable SSL verification (not recommended)
export PYTHONHTTPSVERIFY=0
```

### API Authentication

#### Issue: "Invalid API key" or "Unauthorized"
**Solution:**
```bash
# Verify API key format
echo "NCBI key should be: 1234567890abcdef1234567890abcdef12345678"

# Test API key manually
curl -H "api-key: your_key" "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=protein&term=P53_HUMAN"

# Regenerate key if needed
# Visit: https://www.ncbi.nlm.nih.gov/account/
```

## Data Collection Errors

### Invalid Identifiers

#### Issue: "Identifier not found" or "Invalid accession"
```bash
ERROR: P0463X not found in UniProt
```

**Solution:**
```bash
# Verify identifier format
# UniProt: P04637, Q9Y6K5 (not P0463X)
# PDB: 1A3N, 2AC0 (4 characters)
# Gene symbols: TP53, BRCA1 (exact case matters)

# Search for correct identifier
biocollect collect uniprot-search "P53" --limit 5

# Use alternative identifier types
biocollect collect ensembl TP53 --id-type symbol
```

### Data Format Issues

#### Issue: "JSON decode error" or "Malformed response"
**Solution:**
```bash
# Enable debug logging
biocollect --log-level DEBUG collect uniprot P04637

# Try alternative format
biocollect collect uniprot P04637 --format xml

# Clear cache if corrupted
rm -rf ./cache/*
```

### Sequence Validation Errors

#### Issue: "Invalid protein sequence"
```python
ValidationError: Sequence contains invalid characters
```

**Solution:**
```python
# Check sequence content
from src.utils.validation import SequenceValidator

sequence = "MVLSPADKTNVKAAW"
is_valid = SequenceValidator.is_valid_protein_sequence(sequence)

# Clean sequence if needed
cleaned = SequenceValidator.clean_protein_sequence(sequence)
```

### Feature Extraction Failures

#### Issue: "Failed to extract protein features"
**Solution:**
```bash
# Collect without features first
biocollect collect uniprot P04637 --no-features

# Then add features separately using Python API
python -c "
from src.collectors.uniprot_collector import UniProtCollector
collector = UniProtCollector()
data = collector.collect_single('P04637', include_features=True)
"
```

## Performance Issues

### Slow Database Queries

#### Issue: Very slow protein searches
**Solution:**
```python
# Add database indexes
from sqlalchemy import Index
from src.models.protein import Protein

# Create indexes for common searches
Index('ix_protein_gene_organism', Protein.gene_name, Protein.organism)
Index('ix_protein_sequence_length', Protein.sequence_length)

# For PostgreSQL, analyze tables
# ANALYZE proteins;
# ANALYZE structures;
```

### Memory Issues

#### Issue: "Memory Error" during large collections
**Solution:**
```python
# Process in smaller batches
def process_large_list(items, batch_size=100):
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        # Process batch
        yield batch

# Use streaming for large queries
from src.models.base import get_db
with next(get_db()) as db:
    for protein in db.query(Protein).yield_per(100):
        process_protein(protein)
```

### Disk Space Issues

#### Issue: "No space left on device"
**Solution:**
```bash
# Check disk usage
df -h
du -sh ./data/

# Clean up cache
rm -rf ./cache/*
rm -rf ./data/raw/*.pdb

# Move data to larger disk
export RAW_DATA_DIR=/mnt/large_disk/biocollect/raw
```

### Network Timeouts

#### Issue: Frequent timeout errors
**Solution:**
```bash
# Increase timeouts in .env
echo "API_TIMEOUT=120" >> .env
echo "API_MAX_RETRIES=5" >> .env

# Reduce concurrent requests
echo "MAX_CONCURRENT_REQUESTS=2" >> .env
```

## Platform-Specific Issues

### Windows Issues

#### Issue: Path separator problems
```python
WindowsError: [Error 3] The system cannot find the path specified
```

**Solution:**
```python
# Use pathlib for cross-platform paths
from pathlib import Path

# Instead of: "./data/raw/file.txt"
# Use: Path("data") / "raw" / "file.txt"

# Or normalize paths
import os
path = os.path.normpath("./data/raw/file.txt")
```

#### Issue: Long path names (Windows)
**Solution:**
```bash
# Enable long paths in Windows 10/11
# Run as Administrator:
# New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Or use shorter paths
export RAW_DATA_DIR=C:\data
```

### macOS Issues

#### Issue: "Operation not permitted" (macOS Catalina+)
**Solution:**
```bash
# Grant full disk access to Terminal
# System Preferences > Security & Privacy > Privacy > Full Disk Access
# Add Terminal.app

# Or use different location
mkdir ~/biocollect_data
export RAW_DATA_DIR=~/biocollect_data
```

### Linux Issues

#### Issue: "Permission denied" for system directories
**Solution:**
```bash
# Don't install to system directories
pip install --user -e .

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Performance Optimization

### Slow Collections

#### Diagnosis
```python
# Profile collection performance
import time
from src.collectors.uniprot_collector import UniProtCollector

collector = UniProtCollector()
start_time = time.time()

protein = collector.collect_single("P04637")
collection_time = time.time() - start_time

print(f"Collection took {collection_time:.2f} seconds")
```

#### Solutions
```bash
# Use API keys for better rate limits
echo "UNIPROT_API_KEY=your_key" >> .env
echo "NCBI_API_KEY=your_key" >> .env

# Optimize database
# For PostgreSQL
echo "VACUUM ANALYZE;" | psql biocollect

# For SQLite
sqlite3 biocollect.db "VACUUM; ANALYZE;"

# Use connection pooling
echo "DATABASE_POOL_SIZE=10" >> .env
```

## Debugging Tips

### Enable Debug Logging
```bash
# Command line
biocollect --log-level DEBUG --log-file debug.log collect uniprot P04637

# Python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.collectors.uniprot_collector import UniProtCollector
collector = UniProtCollector()
data = collector.collect_single("P04637")
```

### Inspect API Responses
```python
# Manually test API calls
from src.api.uniprot import UniProtClient

client = UniProtClient()
response = client.get_entry("P04637")
print(response)
```

### Database Inspection
```python
# Check database contents
from src.models.base import get_db
from src.models.protein import Protein

with next(get_db()) as db:
    count = db.query(Protein).count()
    print(f"Total proteins: {count}")
    
    # Recent entries
    recent = db.query(Protein).order_by(
        Protein.created_at.desc()
    ).limit(5).all()
    
    for protein in recent:
        print(f"{protein.accession}: {protein.protein_name}")
```

## Common Error Messages

### "ModuleNotFoundError"
```python
ModuleNotFoundError: No module named 'src'
```
**Solution:**
```bash
# Ensure you're in the project directory
cd datacollect2bionmi
pip install -e .
```

### "ImportError: cannot import name"
```python
ImportError: cannot import name 'UniProtCollector'
```
**Solution:**
```python
# Check import path
from src.collectors.uniprot_collector import UniProtCollector

# Not: from collectors.uniprot_collector import UniProtCollector
```

### "AttributeError" for database models
```python
AttributeError: 'Protein' object has no attribute 'features'
```
**Solution:**
```python
# Ensure relationships are properly loaded
from sqlalchemy.orm import joinedload

protein = db.query(Protein).options(
    joinedload(Protein.features)
).filter(Protein.accession == "P04637").first()
```

## Getting Help

### Self-Diagnosis Tools

#### Check System Status
```bash
# Run comprehensive system check
biocollect status --verbose

# Check configuration
biocollect config --validate

# Test all API connections
python -c "
from src.utils.validation import APIValidator
validator = APIValidator()
validator.test_all_apis()
"
```

#### Generate Debug Report
```python
# debug_report.py
import sys
import platform
from src.models.base import get_db
from src.utils.database import get_table_stats

def generate_debug_report():
    report = {
        'system': {
            'python_version': sys.version,
            'platform': platform.platform(),
            'architecture': platform.architecture()
        },
        'database': {},
        'configuration': {}
    }
    
    # Database status
    try:
        with next(get_db()) as db:
            report['database'] = get_table_stats(db)
    except Exception as e:
        report['database'] = {'error': str(e)}
    
    # Configuration
    import os
    config_vars = [
        'DATABASE_URL', 'UNIPROT_API_KEY', 'NCBI_API_KEY',
        'LOG_LEVEL', 'RAW_DATA_DIR'
    ]
    
    for var in config_vars:
        value = os.getenv(var)
        report['configuration'][var] = 'SET' if value else 'NOT SET'
    
    return report

# Generate and display report
report = generate_debug_report()
import json
print(json.dumps(report, indent=2))
```

### When to Seek Help

Contact support when you encounter:

1. **Reproducible crashes** that occur with specific commands
2. **Data corruption** or inconsistent results
3. **Performance issues** that persist after optimization
4. **Installation failures** on supported platforms
5. **API authentication issues** with valid credentials

### How to Report Issues

When reporting issues, include:

1. **Complete error message** with stack trace
2. **Command or code** that caused the error
3. **System information** (OS, Python version)
4. **Configuration** (without sensitive API keys)
5. **Debug log** if available
6. **Steps to reproduce** the issue

### Community Resources

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share solutions
- **Documentation**: Check latest version online
- **API Documentation**: Refer to upstream database APIs

### Quick Fix Checklist

Before seeking help, try these common fixes:

- [ ] Restart your terminal/IDE
- [ ] Activate virtual environment
- [ ] Update to latest version: `pip install -e . --upgrade`
- [ ] Clear cache: `rm -rf ./cache/*`
- [ ] Reset database: `rm biocollect.db && biocollect init`
- [ ] Check network connectivity
- [ ] Verify API keys are valid
- [ ] Review recent configuration changes
- [ ] Check disk space: `df -h`
- [ ] Review error logs: `tail -f biocollect.log`