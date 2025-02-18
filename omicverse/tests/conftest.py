import pytest


def pytest_addoption(parser):
    """Docstring for pytest_addoption."""
    parser.addoption(
        "--accelerator",
        action="store",
        default="cpu",
        help="Option to specify which accelerator to use for tests.",
    )
    parser.addoption(
        "--cuml",
        action="store",
        default=False,
        help="Option to specify whether cuml is used.",
    )
    parser.addoption(
        "--private",
        action="store",
        default=False,
        help="Option to specify whether huggingface credentials are available.",
    )


@pytest.fixture(scope="session")
def accelerator(request):
    """Defines whether CPU or GPU is used for tests. Default CPU."""
    return request.config.getoption("--accelerator")


@pytest.fixture(scope="session")
def cuml(request):
    """Defines whether cuml is used for tests. Default False."""
    return request.config.getoption("--cuml")


@pytest.fixture(scope="session")
def private(request):
    """Defines whether huggingface credentials are available. Default False."""
    return request.config.getoption("--private")
