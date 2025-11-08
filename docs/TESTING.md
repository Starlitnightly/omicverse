# OmicVerse Testing Guide

## Overview

This document provides comprehensive testing instructions for the OmicVerse agent system, including test coverage goals, offline testing guidelines, and integration test procedures.

## Table of Contents

1. [Test Structure](#test-structure)
2. [Running Tests](#running-tests)
3. [Test Coverage](#test-coverage)
4. [Offline Testing](#offline-testing)
5. [Integration Tests](#integration-tests)
6. [Adding New Tests](#adding-new-tests)
7. [Continuous Integration](#continuous-integration)

---

## Test Structure

### Test Organization

Tests are organized in the `tests/` directory with the following structure:

```
tests/
├── utils/                           # Agent and utility tests
│   ├── test_agent_backend_providers.py    # Provider integration tests
│   ├── test_agent_backend_streaming.py    # Streaming API tests
│   ├── test_agent_backend_usage.py        # Token usage tracking tests
│   ├── test_agent_initialization.py       # Agent init tests
│   ├── test_smart_agent.py               # Smart agent functionality
│   ├── test_model_normalization.py       # Model ID alias tests
│   └── test_skill_instruction_formatter.py # Skill formatting tests
├── bulk/                            # Bulk RNA-seq analysis tests
├── single/                          # Single-cell analysis tests
├── llm/                            # LLM-specific tests
└── others/                         # Miscellaneous tests
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes in isolation
2. **Integration Tests**: Test interactions between components
3. **Provider Tests**: Test LLM provider integrations
4. **Streaming Tests**: Test async streaming functionality
5. **End-to-End Tests**: Test complete user workflows

---

## Running Tests

### Prerequisites

Install test dependencies:

```bash
# Install core dependencies
pip install pytest pytest-cov pytest-asyncio

# Install omicverse with test dependencies
pip install -e ".[tests]"
```

### Basic Test Execution

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/utils/test_agent_backend_usage.py
```

Run specific test class:
```bash
pytest tests/utils/test_agent_backend_usage.py::TestUsageDataclass
```

Run specific test function:
```bash
pytest tests/utils/test_agent_backend_usage.py::TestUsageDataclass::test_usage_dataclass_creation
```

### Verbose Output

```bash
pytest -v                          # Verbose test names
pytest -vv                         # Very verbose with full diffs
pytest -s                          # Show print statements
pytest -v -s                       # Combine verbose + print output
```

### Async Tests

Async tests use `pytest-asyncio`:

```bash
# Run async tests
pytest tests/utils/test_agent_backend_streaming.py -v

# Run with asyncio debug mode
pytest tests/utils/test_agent_backend_streaming.py -v --asyncio-mode=auto
```

---

## Test Coverage

### Coverage Goals

**Target: >80% coverage for agent modules**

Current coverage (as of 2025-01-08):
- `agent_backend.py`: 45% (target: 85%)
- `model_config.py`: 71% (target: 90%)
- `smart_agent.py`: 0% (target: 80%)
- `skill_registry.py`: 0% (target: 75%)

### Running Coverage Tests

#### Method 1: pytest-cov

```bash
# Run with coverage report
pytest tests/utils/ --cov=omicverse/utils --cov-report=term-missing

# Generate HTML report
pytest tests/utils/ --cov=omicverse/utils --cov-report=html

# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

#### Method 2: coverage.py

```bash
# Run tests with coverage
coverage run -m pytest tests/utils/test_agent_backend_usage.py

# Generate report
coverage report --include="omicverse/utils/agent_backend.py,omicverse/utils/model_config.py"

# Generate HTML report
coverage html --include="omicverse/utils/*.py"

# Generate JSON report
coverage json
```

### Coverage Reports

#### Terminal Report (term-missing)

Shows missing line numbers:
```
Name                               Stmts   Miss  Cover   Missing
----------------------------------------------------------------
omicverse/utils/agent_backend.py     605    332    45%   123-145, 200-250, ...
omicverse/utils/model_config.py      114     33    71%   50-65, 180-190
----------------------------------------------------------------
TOTAL                                719    365    49%
```

#### HTML Report

Interactive HTML report with line-by-line coverage:
```bash
coverage html
open htmlcov/index.html
```

Features:
- Color-coded coverage (green = covered, red = missed)
- Line-by-line execution counts
- Branch coverage visualization
- Sortable by coverage percentage

#### JSON Report

Machine-readable format for CI/CD:
```bash
coverage json
cat coverage.json | jq '.totals.percent_covered'
```

### Improving Coverage

#### Areas Needing Tests

1. **agent_backend.py** (45% → 85% target):
   - Error handling paths
   - Retry logic for different error types
   - HTTP fallback mechanisms
   - Provider-specific edge cases

2. **model_config.py** (71% → 90% target):
   - API key validation edge cases
   - Model normalization for all aliases
   - Configuration error handling

3. **smart_agent.py** (0% → 80% target):
   - Code extraction with complex patterns
   - Reflection mechanism
   - Result review mechanism
   - Skill matching (LLM and algorithmic)
   - End-to-end query execution

4. **skill_registry.py** (0% → 75% target):
   - Skill loading and caching
   - Progressive disclosure logic
   - Skill metadata extraction

#### Test Priority

**High Priority** (blocking 80% coverage):
1. Add end-to-end tests for `smart_agent.py`
2. Test reflection mechanism
3. Test code extraction edge cases
4. Test error handling in `agent_backend.py`

**Medium Priority**:
1. Test all provider streaming paths
2. Test skill registry caching
3. Test model normalization for all 84 models

**Low Priority** (nice to have):
1. Test edge cases for rare error types
2. Test with actual API keys (optional, expensive)

---

## Offline Testing

### Principle

**All tests must run offline without internet or API keys.**

This ensures:
- Fast test execution
- No API costs during testing
- Reproducible test results
- CI/CD compatibility

### Mocking Strategy

#### 1. Mock LLM API Calls

Use `unittest.mock` to mock external API calls:

```python
from unittest.mock import patch, MagicMock
import pytest
from omicverse.utils.agent_backend import OmicVerseLLMBackend

def test_openai_call_offline():
    """Test OpenAI call without hitting real API"""
    backend = OmicVerseLLMBackend(
        model="openai/gpt-4o",
        api_key="fake-key-for-testing"
    )

    # Mock the OpenAI SDK
    with patch('openai.OpenAI') as mock_openai:
        # Create mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Mocked response"))
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )

        # Configure mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Make the call
        messages = [{"role": "user", "content": "Test query"}]
        response, usage = backend.call(messages)

        # Assertions
        assert response == "Mocked response"
        assert usage.total_tokens == 150
        assert mock_client.chat.completions.create.called
```

#### 2. Mock Streaming Responses

```python
@pytest.mark.asyncio
async def test_streaming_offline():
    """Test streaming without real API"""
    backend = OmicVerseLLMBackend(
        model="openai/gpt-4o",
        api_key="fake-key"
    )

    # Mock streaming response
    async def mock_stream():
        chunks = ["Hello", " ", "world", "!"]
        for chunk in chunks:
            mock_chunk = MagicMock()
            mock_chunk.choices = [
                MagicMock(delta=MagicMock(content=chunk))
            ]
            yield mock_chunk

    with patch('openai.AsyncOpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream()
        mock_openai.return_value = mock_client

        # Collect stream
        result = []
        async for chunk, usage in backend.stream([
            {"role": "user", "content": "Test"}
        ]):
            if chunk:
                result.append(chunk)

        assert ''.join(result) == "Hello world!"
```

#### 3. Mock File I/O

```python
from unittest.mock import mock_open

def test_skill_loading_offline():
    """Test skill loading without file system access"""
    skill_content = """
    # Skill: single-cell-preprocessing
    Description: Preprocess single-cell data
    """

    with patch('builtins.open', mock_open(read_data=skill_content)):
        skill = load_skill("single-cell-preprocessing")
        assert skill.name == "single-cell-preprocessing"
```

### Environment Variables for Testing

```python
import os

@pytest.fixture(autouse=True)
def offline_mode():
    """Force offline mode for all tests"""
    original_env = os.environ.copy()

    # Set offline flags
    os.environ['OMICVERSE_OFFLINE'] = '1'
    os.environ['NO_INTERNET'] = '1'

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
```

### Testing with Mock Data

Create fixtures for common test data:

```python
# tests/conftest.py

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData

@pytest.fixture
def mock_adata():
    """Create mock AnnData object for testing"""
    n_obs = 100
    n_vars = 50

    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame({
        'cell_type': ['T cell'] * 50 + ['B cell'] * 50,
        'batch': ['batch1'] * 30 + ['batch2'] * 70
    }, index=[f'cell_{i}' for i in range(n_obs)])
    var = pd.DataFrame({
        'gene_name': [f'Gene_{i}' for i in range(n_vars)]
    }, index=[f'gene_{i}' for i in range(n_vars)])

    return AnnData(X=X, obs=obs, var=var)

@pytest.fixture
def mock_llm_response():
    """Mock LLM response with code"""
    return """
    Here's the code to preprocess your data:

    ```python
    import omicverse as ov

    # Preprocess
    adata = ov.pp.qc(adata)
    adata = ov.pp.normalize(adata)
    ```

    This will normalize your data.
    """
```

### Offline Test Markers

```python
# Mark tests as offline
@pytest.mark.offline
def test_code_extraction():
    """This test runs without network access"""
    pass

# Mark tests requiring internet
@pytest.mark.online
@pytest.mark.skipif(
    os.getenv('OMICVERSE_OFFLINE') == '1',
    reason="Requires internet access"
)
def test_real_api():
    """This test requires real API access"""
    pass
```

Run only offline tests:
```bash
pytest -m offline
```

Skip online tests:
```bash
pytest -m "not online"
```

---

## Integration Tests

### What are Integration Tests?

Integration tests verify that multiple components work together correctly, testing:
- Agent → LLM Backend → Provider communication
- Code Generation → Extraction → Execution pipeline
- Reflection → Error Correction workflow
- Streaming event flow

### Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with detailed logging
pytest tests/integration/ -v -s --log-cli-level=DEBUG
```

### Example Integration Tests

#### 1. End-to-End Agent Query

```python
# tests/integration/test_agent_e2e.py

import pytest
from unittest.mock import patch
from omicverse.utils.smart_agent import OmicVerseAgent

@pytest.mark.integration
def test_complete_query_workflow(mock_adata, mock_llm_response):
    """Test complete query from user input to result"""

    agent = OmicVerseAgent(
        model="openai/gpt-4o",
        api_key="fake-key"
    )

    # Mock LLM backend
    with patch.object(agent.llm_backend, 'call') as mock_call:
        mock_call.return_value = (mock_llm_response, MagicMock())

        # Execute query
        result = agent.query(
            "Preprocess my single-cell data",
            adata=mock_adata
        )

        # Assertions
        assert result['success'] is True
        assert 'adata' in result
        assert mock_call.called
```

#### 2. Reflection Mechanism Integration

```python
@pytest.mark.integration
def test_reflection_on_error(mock_adata):
    """Test that reflection corrects code errors"""

    agent = OmicVerseAgent(model="openai/gpt-4o", api_key="fake-key")

    # First response has error
    bad_code = """```python
    import omicverse as ov
    adata = undefined_function()  # This will fail
    ```"""

    # Second response corrects error
    good_code = """```python
    import omicverse as ov
    adata = ov.pp.qc(adata)
    ```"""

    with patch.object(agent.llm_backend, 'call') as mock_call:
        # First call returns bad code, second returns correction
        mock_call.side_effect = [
            (bad_code, MagicMock()),
            (good_code, MagicMock())
        ]

        result = agent.query(
            "Preprocess my data",
            adata=mock_adata,
            max_reflections=3
        )

        # Should have called LLM twice (original + reflection)
        assert mock_call.call_count == 2
        assert result['success'] is True
```

#### 3. Streaming Event Flow

```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_streaming_event_flow(mock_adata):
    """Test complete streaming event sequence"""

    agent = OmicVerseAgent(model="openai/gpt-4o", api_key="fake-key")

    # Mock streaming response
    async def mock_stream(*args, **kwargs):
        chunks = ["import ", "omicverse ", "as ", "ov"]
        for chunk in chunks:
            yield chunk, None
        yield "", MagicMock(total_tokens=100)

    with patch.object(agent.llm_backend, 'stream', mock_stream):
        events = []
        async for event in agent.stream_async(
            "Preprocess my data",
            adata=mock_adata
        ):
            events.append(event)

        # Verify event sequence
        event_types = [e['event_type'] for e in events]
        assert 'skill_match' in event_types
        assert 'llm_chunk' in event_types
        assert 'code' in event_types
        assert 'result' in event_types
        assert 'usage' in event_types
```

### Integration Test Best Practices

1. **Test Real Workflows**: Mimic actual user interactions
2. **Mock External Calls**: Keep tests offline and fast
3. **Verify Event Ordering**: Ensure correct sequence
4. **Test Error Paths**: Include failure scenarios
5. **Use Fixtures**: Reuse common test data
6. **Add Logging**: Help debug failures with detailed logs

### Integration Test Checklist

- [ ] Agent initialization with various configurations
- [ ] Query execution with valid input
- [ ] Query execution with invalid input
- [ ] Reflection mechanism triggered by errors
- [ ] Result review validation
- [ ] Streaming event sequence
- [ ] Multi-provider compatibility
- [ ] Token usage tracking across workflow
- [ ] Skill matching and loading
- [ ] Code extraction from various formats

---

## Adding New Tests

### Test File Template

```python
# tests/utils/test_new_feature.py

"""
Tests for new feature functionality.

This module tests:
- Feature A
- Feature B
- Edge cases for Feature C
"""

import pytest
from unittest.mock import patch, MagicMock
from omicverse.utils.new_module import NewFeature


class TestNewFeature:
    """Test suite for NewFeature class"""

    def test_basic_functionality(self):
        """Test basic feature operation"""
        feature = NewFeature()
        result = feature.process("input")
        assert result == "expected_output"

    def test_error_handling(self):
        """Test error handling"""
        feature = NewFeature()
        with pytest.raises(ValueError):
            feature.process(None)

    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operation"""
        feature = NewFeature()
        result = await feature.process_async("input")
        assert result is not None

    @pytest.mark.parametrize("input,expected", [
        ("test1", "output1"),
        ("test2", "output2"),
        ("test3", "output3"),
    ])
    def test_parametrized(self, input, expected):
        """Test multiple input cases"""
        feature = NewFeature()
        assert feature.process(input) == expected
```

### Writing Good Tests

#### 1. Test Naming

Use descriptive names that explain what is being tested:

```python
# Good
def test_openai_sdk_retries_on_rate_limit():
    pass

# Bad
def test_retry():
    pass
```

#### 2. Test Structure (AAA Pattern)

```python
def test_feature():
    # Arrange: Set up test data
    feature = NewFeature()
    input_data = "test"

    # Act: Execute the code being tested
    result = feature.process(input_data)

    # Assert: Verify the result
    assert result == "expected"
```

#### 3. Test One Thing

```python
# Good: Test one behavior
def test_validates_email_format():
    assert is_valid_email("test@example.com") is True

def test_rejects_invalid_email():
    assert is_valid_email("invalid") is False

# Bad: Test multiple behaviors
def test_email():
    assert is_valid_email("test@example.com") is True
    assert is_valid_email("invalid") is False
    assert send_email("test@example.com", "subject", "body") is True
```

#### 4. Use Fixtures

```python
@pytest.fixture
def configured_agent():
    """Fixture providing a configured agent"""
    return OmicVerseAgent(
        model="openai/gpt-4o",
        temperature=0.1,
        api_key="test-key"
    )

def test_with_fixture(configured_agent):
    """Use the fixture in tests"""
    result = configured_agent.query("test", adata)
    assert result is not None
```

### Testing Async Code

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async function"""
    result = await async_operation()
    assert result is not None

@pytest.mark.asyncio
async def test_async_generator():
    """Test async generator"""
    results = []
    async for item in async_generator():
        results.append(item)
    assert len(results) > 0
```

### Mocking Best Practices

```python
# Mock at the right level
with patch('omicverse.utils.agent_backend.OpenAI') as mock:
    # Configure mock
    mock.return_value.chat.completions.create.return_value = response

# Use side_effect for sequences
mock.side_effect = [response1, response2, Exception("error")]

# Use MagicMock for complex objects
mock_obj = MagicMock()
mock_obj.method.return_value = value
mock_obj.attribute = "value"

# Verify mock calls
assert mock.called
assert mock.call_count == 2
mock.assert_called_once_with(arg1, arg2)
```

---

## Continuous Integration

### CI/CD Pipeline

Example GitHub Actions workflow:

```yaml
# .github/workflows/test.yml

name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[tests]"
        pip install pytest-cov

    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=omicverse --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

    - name: Check coverage threshold
      run: |
        coverage report --fail-under=80
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml

repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

### Coverage Badges

Add to README.md:

```markdown
[![Coverage](https://codecov.io/gh/your-org/omicverse/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/omicverse)
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'omicverse'

# Solution: Install in editable mode
pip install -e .
```

#### 2. Async Tests Not Running

```bash
# Error: async def functions are not natively supported

# Solution: Install pytest-asyncio
pip install pytest-asyncio
```

#### 3. Coverage Not Detecting Modules

```bash
# Error: Module was never imported

# Solution: Use coverage run instead of pytest --cov
coverage run -m pytest tests/
coverage report
```

#### 4. Tests Hitting Real APIs

```bash
# Error: Tests making real API calls

# Solution: Check mocking
# Ensure patches are at the right level
# Use offline markers
```

### Debug Mode

```bash
# Run with debugger
pytest --pdb

# Drop into debugger on failure
pytest --pdb --maxfail=1

# Show local variables on failure
pytest -l

# Verbose output
pytest -vv -s
```

---

## Test Coverage Summary

### Current Coverage (as of 2025-01-08)

| Module | Statements | Coverage | Target | Priority |
|--------|-----------|----------|--------|----------|
| agent_backend.py | 605 | 45% | 85% | High |
| model_config.py | 114 | 71% | 90% | Medium |
| smart_agent.py | 521 | 0% | 80% | High |
| skill_registry.py | 321 | 0% | 75% | High |
| registry.py | 243 | 0% | 70% | Low |

### Test Files Status

- ✅ `test_agent_backend_usage.py` - 14 tests passing
- ✅ `test_model_normalization.py` - 2 tests passing
- ✅ `test_skill_instruction_formatter.py` - 2 tests passing
- ⚠️ `test_agent_backend_providers.py` - Requires full dependencies
- ⚠️ `test_agent_backend_streaming.py` - Requires full dependencies
- ⚠️ `test_smart_agent.py` - Needs expansion
- ⚠️ `test_agent_initialization.py` - Needs expansion

### Next Steps

1. **Immediate** (to reach 80%):
   - Add end-to-end tests for `smart_agent.py`
   - Test code extraction edge cases
   - Test reflection mechanism
   - Test error handling paths in `agent_backend.py`

2. **Short-term** (to reach 90%):
   - Add integration tests
   - Test all provider streaming paths
   - Test skill registry thoroughly
   - Add fuzzing tests for code extraction

3. **Long-term** (maintain >90%):
   - Add property-based tests
   - Add performance regression tests
   - Add load tests for concurrent requests

---

## Resources

### Documentation

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [coverage.py](https://coverage.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

### Best Practices

- [Test-Driven Development](https://testdriven.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Mock Object Patterns](https://martinfowler.com/articles/mocksArentStubs.html)

---

**Document Version:** 1.0
**Last Updated:** 2025-01-08
**Maintainer:** OmicVerse Development Team
