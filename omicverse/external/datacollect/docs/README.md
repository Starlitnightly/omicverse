# BioinformaticsDataCollector Documentation

Welcome to the documentation for BioinformaticsDataCollector, a comprehensive tool for collecting and managing biological data from major bioinformatics databases.

## Documentation Contents

### Getting Started
- **[Main README](../README.md)** - Project overview and quick start guide
- **[Tutorial](TUTORIAL.md)** - Step-by-step guide with examples
- **[Quick Reference](QUICK_REFERENCE.md)** - Common commands and code snippets

### Detailed Guides
- **[API Documentation](API.md)** - Complete API reference for all modules
- **[Configuration Guide](CONFIGURATION.md)** - All configuration options explained
- **[Developer Guide](DEVELOPER.md)** - Contributing and extending the project

## Quick Links

### For Users
1. Start with the [Tutorial](TUTORIAL.md) to learn basic usage
2. Use the [Quick Reference](QUICK_REFERENCE.md) for common tasks
3. Check [Configuration Guide](CONFIGURATION.md) for customization

### For Developers
1. Read the [Developer Guide](DEVELOPER.md) for contribution guidelines
2. Refer to [API Documentation](API.md) for detailed method references
3. Check the source code for implementation details

## Key Features

- **Multi-Database Support**: UniProt, PDB, NCBI, Ensembl, KEGG
- **Data Validation**: Automatic validation of biological data
- **Local Storage**: SQLite/PostgreSQL with SQLAlchemy ORM
- **CLI Interface**: User-friendly command-line tools
- **Python API**: Programmatic access to all functionality
- **Batch Processing**: Efficient handling of large datasets
- **Rate Limiting**: Respectful API usage with built-in limits

## Example Usage

### Command Line
```bash
# Initialize database
biocollect init

# Collect protein data
biocollect collect uniprot P04637

# Collect structure data
biocollect collect pdb 1A3N --download
```

### Python API
```python
from src.collectors.uniprot_collector import UniProtCollector

collector = UniProtCollector()
protein = collector.process_and_save("P04637")
print(f"Collected: {protein.protein_name}")
```

## Support

- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/datacollect2bionmi/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/yourusername/datacollect2bionmi/discussions)
- **Contributing**: See [Developer Guide](DEVELOPER.md) for guidelines

## License

This project is licensed under the MIT License. See the LICENSE file for details.