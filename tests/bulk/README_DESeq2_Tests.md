# DESeq2 Memory Optimization Tests

This directory contains comprehensive test suites for verifying DESeq2 memory optimization functionality in OmicVerse, specifically addressing Issue #347.

## Test Files Overview

### 1. `test_deseq2_basic.py`
**Purpose**: Tests core DESeq2 functionality and ensures basic operations work correctly.

**Key Test Cases**:
- pyDEG initialization and basic functionality
- Drop duplicates index functionality
- DEG analysis with t-test and DESeq2 methods
- Fold change threshold setting
- Boxplot generation
- GSEA ranking generation
- Error handling for invalid inputs

**Usage**:
```bash
pytest tests/bulk/test_deseq2_basic.py -v
```

### 2. `test_issue_347_deseq2_memory.py`
**Purpose**: Specifically addresses the memory explosion issue reported in Issue #347 with large single-cell datasets (31K+ cells).

**Key Test Cases**:
- **Original Issue Scenario**: Reproduces the exact scenario from Issue #347
- **Memory Usage Progression**: Tracks memory at each analysis step
- **Large Group Handling**: Tests handling of large treatment/control groups
- **Memory Cleanup**: Verifies proper memory cleanup after analysis
- **Error Handling**: Tests edge cases that might cause memory issues
- **Performance Benchmark**: Ensures optimization doesn't hurt performance
- **Memory Optimization Fallback**: Tests fallback mechanisms

**Usage**:
```bash
pytest tests/bulk/test_issue_347_deseq2_memory.py -v -s
```

### 3. `test_deseq2_memory_optimization.py`
**Purpose**: Comprehensive test suite for memory optimization features across various scenarios.

**Key Test Cases**:
- **Memory Monitoring**: Tests memory usage tracking functionality
- **Memory Efficient Parameters**: Tests `memory_efficient` and `chunk_size` parameters
- **Auto-fallback**: Tests automatic fallback to t-test for extremely large datasets
- **Memory Optimization Suggestions**: Tests memory optimization recommendation system
- **Subsampling**: Tests intelligent subsampling for large groups
- **Memory Estimation**: Tests memory requirement prediction
- **DESeq2-specific Optimizations**: Tests DESeq2-specific memory optimizations
- **Integration Tests**: End-to-end testing with large datasets

**Usage**:
```bash
pytest tests/bulk/test_deseq2_memory_optimization.py -v
```

## Expected Memory Optimization Features

These tests verify the following features that should be implemented to resolve Issue #347:

### 1. **Memory Monitoring**
- Real-time memory usage tracking during analysis
- Warnings when memory usage approaches system limits
- Memory usage reporting at each analysis step

### 2. **Memory-Efficient Processing**
- `memory_efficient=True` parameter for DESeq2 analysis
- Chunked processing for large datasets
- Automatic memory cleanup between processing steps

### 3. **Intelligent Subsampling**
- Automatic subsampling when groups exceed reasonable size (e.g., >5K cells per group)
- Maintains statistical power while reducing memory requirements
- Configurable subsampling parameters

### 4. **Auto-fallback Mechanisms**
- Automatic fallback to t-test for extremely large datasets (>50K cells)
- User notification when fallback occurs
- Configurable fallback thresholds

### 5. **Memory Optimization Suggestions**
- `get_memory_optimization_suggestions()` method
- Recommendations based on dataset size and available memory
- Guidance on parameter tuning for memory efficiency

### 6. **Memory Requirement Estimation**
- `estimate_memory_requirements()` method
- Prediction of memory usage before analysis starts
- Warning system for insufficient memory

## Running the Tests

### Prerequisites
```bash
pip install pytest psutil numpy pandas omicverse pydeseq2
```

### Run All DESeq2 Tests
```bash
pytest tests/bulk/test_deseq2_*.py -v
```

### Run Specific Test Categories
```bash
# Basic functionality tests
pytest tests/bulk/test_deseq2_basic.py -v

# Issue #347 specific tests  
pytest tests/bulk/test_issue_347_deseq2_memory.py -v -s

# Comprehensive memory optimization tests
pytest tests/bulk/test_deseq2_memory_optimization.py -v
```

### Run Tests with Memory Monitoring
```bash
pytest tests/bulk/test_issue_347_deseq2_memory.py::TestIssue347DESeq2Memory::test_memory_usage_progression -v -s
```

## Test Data

The tests use simulated datasets that mirror the characteristics of the original Issue #347:

- **Large Single-Cell Dataset**: 1K-2K cells × 500-1K genes (scaled down for CI)
- **Realistic Count Data**: Negative binomial distribution to simulate real scRNA-seq
- **Cell Metadata**: Includes cell types, treatment labels, and batch information
- **Large Groups**: Unbalanced treatment/control groups to test edge cases

## Expected Behavior

### Without Memory Optimization
- Tests may fail with MemoryError on large datasets
- Memory usage increases significantly during DESeq2 analysis
- No fallback mechanisms for memory issues

### With Memory Optimization
- Tests should pass even with large datasets
- Memory usage should be controlled and predictable
- Automatic fallbacks should trigger for extremely large datasets
- Memory optimization suggestions should be provided
- Performance should remain reasonable

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Install `psutil` for memory monitoring
2. **pydeseq2 Version**: Ensure pydeseq2 > 0.3 is installed
3. **Memory Limitations**: Tests may be skipped on systems with <8GB RAM

### Debug Mode
Run tests with additional verbosity:
```bash
pytest tests/bulk/test_issue_347_deseq2_memory.py -v -s --tb=long
```

## Integration with Issue #347

These tests directly address the memory explosion problem reported in Issue #347:

> "31002细胞x3000个基因时出现问题"
> "⏰ Start to create DeseqDataSet..."  
> "Fitting MAP dispersions... [memory explosion]"

The tests verify that:
1. Large datasets can be processed without memory explosion
2. Memory usage is monitored and controlled
3. Fallback mechanisms work when memory is insufficient
4. The fix maintains statistical accuracy and performance

## Contributing

When adding new memory optimization features:

1. Add corresponding tests to verify the functionality
2. Update this README with new test descriptions
3. Ensure tests cover both success and failure scenarios
4. Include performance benchmarks for new optimizations