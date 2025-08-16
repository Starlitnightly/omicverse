# DataCollect Test Suite

**Comprehensive testing framework for OmicVerse DataCollect external module**

## 🧪 Available Test Scripts

### 1. **test_suite_summary.py** - Test Orchestration
**Start here for overview and intelligent test selection**

```bash
python test_suite_summary.py                    # Show available tests and environment
python test_suite_summary.py --run-available    # Run tests that work in current environment
```

### 2. **test_ov_datacollect_network.py** - Network Connectivity ✅
**Tests API connectivity to 22+ biological databases - ALWAYS WORKS**

```bash
python test_ov_datacollect_network.py                          # Test all APIs
python test_ov_datacollect_network.py --apis UniProt KEGG      # Test specific APIs
python test_ov_datacollect_network.py --timeout 30             # Longer timeout
python test_ov_datacollect_network.py --output report.txt      # Save detailed report
```

**Results**: Tests connectivity to UniProt, PDB, AlphaFold, Ensembl, KEGG, Reactome, GEO, OpenTargets, and more.

### 3. **test_ov_datacollect_quick.py** - Quick Validation
**Fast 8-test verification of basic functionality**

```bash
python test_ov_datacollect_quick.py    # ~10 seconds, requires OmicVerse
```

**Tests**: Import, modules, functions, clients, formats, info, calls, dependencies

### 4. **test_omicverse_datacollect_complete.py** - Comprehensive Testing
**Complete validation of all DataCollect components**

```bash
python test_omicverse_datacollect_complete.py                    # Full test suite
python test_omicverse_datacollect_complete.py --verbose          # Detailed output
python test_omicverse_datacollect_complete.py --category api     # Test API clients only
python test_omicverse_datacollect_complete.py --category formats # Test format converters
```

**Categories**: Main functions, API clients (29), format converters, utilities, integration

### 5. **demo_ov_datacollect_usage.py** - Interactive Learning
**Practical examples and workflow demonstrations**

```bash
python demo_ov_datacollect_usage.py                    # All demos
python demo_ov_datacollect_usage.py --demo basic       # Basic usage only
python demo_ov_datacollect_usage.py --demo integration # OmicVerse integration
python demo_ov_datacollect_usage.py --demo workflow    # Complete workflows
```

**Demos**: Basic collection, integration examples, format conversion, complete workflows

### 6. **test_datacollect_standalone.py** - Independent Testing
**Test DataCollect without full OmicVerse installation**

```bash
python test_datacollect_standalone.py    # Works without OmicVerse
```

## 🚀 Quick Start

### Option 1: Automatic (Recommended)
```bash
python test_suite_summary.py --run-available
```

### Option 2: Network Test (Always Works)
```bash
python test_ov_datacollect_network.py --apis UniProt KEGG Ensembl
```

### Option 3: Full Validation (Requires OmicVerse)
```bash
python test_ov_datacollect_quick.py
python test_omicverse_datacollect_complete.py
```

## 📊 Test Results Interpretation

### Network Test Results
- **100% Success**: All biological databases accessible ✅
- **80%+ Success**: Most APIs working, some may need authentication ⚠️
- **<80% Success**: Network/firewall issues ❌

### OmicVerse Integration Results
- **8/8 Passed**: DataCollect fully functional ✅
- **6-7/8 Passed**: Minor issues, mostly functional ⚠️
- **<6/8 Passed**: Installation or dependency issues ❌

## 🔧 Troubleshooting

### Common Issues

1. **"zarr-python major version > 2 is not supported"**
   ```bash
   pip install "zarr<3.0.0"
   ```

2. **"No module named 'torch'"**
   ```bash
   pip install torch
   ```

3. **Network API failures**
   - Check internet connection
   - Verify firewall settings
   - Some APIs require authentication keys

### Environment Requirements

| Test Script | Requires OmicVerse | Requires Network | Notes |
|-------------|-------------------|------------------|-------|
| `test_suite_summary.py` | No | No | Always works |
| `test_ov_datacollect_network.py` | No | Yes | Always recommended |
| `test_ov_datacollect_quick.py` | Yes | No | Fast validation |
| `test_omicverse_datacollect_complete.py` | Yes | No | Comprehensive |
| `demo_ov_datacollect_usage.py` | Yes | No | Learning/examples |
| `test_datacollect_standalone.py` | No | No | Limited scope |

## 📈 Test Coverage

### APIs Tested (22+)
- **Proteins**: UniProt, PDB, AlphaFold, InterPro, STRING, EMDB
- **Genomics**: Ensembl, ClinVar, dbSNP, gnomAD, GWAS Catalog, UCSC
- **Expression**: GEO, OpenTargets, ReMap
- **Pathways**: KEGG, Reactome, GtoPdb
- **Specialized**: BLAST, JASPAR, IUCN, PRIDE

### Functionality Tested
- ✅ Import and module availability
- ✅ API client instantiation and functionality
- ✅ Format conversion (pandas, AnnData, MuData)
- ✅ OmicVerse integration and compatibility
- ✅ Network connectivity and error handling
- ✅ Configuration and utilities

## 🎯 Best Practices

1. **Start with**: `python test_suite_summary.py`
2. **Always run**: `python test_ov_datacollect_network.py`
3. **For CI/CD**: Use `--output` flags to save reports
4. **For debugging**: Use `--verbose` flags for detailed output
5. **For learning**: Run demo scripts with specific categories

## 📚 Related Documentation

- **Main README**: `README.md`
- **API Reference**: `docs/OMICVERSE_API_REFERENCE.md`
- **Integration Guide**: `docs/OMICVERSE_INTEGRATION_GUIDE.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Configuration**: `docs/CONFIGURATION.md`

---

**This test suite provides complete validation of DataCollect functionality, from basic imports to comprehensive API connectivity testing across 22+ biological databases.**