"""
Shared pytest configuration and fixtures for OmicVerse test suite.

This file contains common fixtures and configuration that are shared across
all test modules in the OmicVerse project.
"""

import asyncio
import importlib
import inspect
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path


@pytest.fixture
def tmp_path_factory_session(tmp_path_factory):
    """
    Provides a session-scoped temporary directory factory.
    Useful for creating temporary files that persist across multiple tests.
    """
    return tmp_path_factory


@pytest.fixture
def random_seed():
    """
    Provides a consistent random seed for reproducible tests.
    """
    return 42


@pytest.fixture
def sample_dataframe(random_seed):
    """
    Creates a simple pandas DataFrame for testing.
    """
    np.random.seed(random_seed)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'label': np.random.choice(['A', 'B', 'C'], 100)
    })


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Provides a temporary directory for test outputs.
    Automatically cleaned up after test completion.
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture(autouse=True)
def reset_random_state(random_seed):
    """
    Automatically reset random state before each test for reproducibility.
    """
    np.random.seed(random_seed)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Fixture to safely mock environment variables in tests.
    Usage:
        def test_something(mock_env_vars):
            mock_env_vars({'API_KEY': 'test_key'})
    """
    def _set_env_vars(env_dict):
        for key, value in env_dict.items():
            monkeypatch.setenv(key, value)
    return _set_env_vars


# Configure pytest to use non-interactive matplotlib backend
@pytest.fixture(scope='session', autouse=True)
def configure_matplotlib():
    """
    Configure matplotlib to use non-interactive backend for all tests.
    This prevents tests from trying to display plots during test runs.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        pass  # matplotlib not installed, skip configuration


# Add custom markers for test categorization
def pytest_configure(config):
    """
    Register custom pytest markers for better test organization.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_network: marks tests that require network access"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests that should run in an event loop"
    )


_has_pytest_asyncio = importlib.util.find_spec("pytest_asyncio") is not None


def pytest_pyfunc_call(pyfuncitem):
    """Run ``@pytest.mark.asyncio`` tests when pytest-asyncio is unavailable.

    The repository originally provided its own minimal asyncio runner because
    pytest-asyncio was not a hard dependency. When the plugin *is* installed we
    defer to its implementation to avoid double-scheduling the coroutine (which
    leads to ``TypeError: An asyncio.Future, a coroutine or an awaitable is
    required``). If the plugin is missing, we fall back to the lightweight
    runner below so async tests still execute correctly.
    """

    if _has_pytest_asyncio:
        # Let pytest-asyncio manage event loop lifecycle
        return None

    if pyfuncitem.get_closest_marker("asyncio"):
        async_fn = pyfuncitem.obj
        fn_params = inspect.signature(async_fn).parameters
        bound_args = {name: pyfuncitem.funcargs[name] for name in fn_params}

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(async_fn(**bound_args))
        finally:
            loop.close()
        return True
