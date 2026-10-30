import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="also run tests marked as slow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test: use --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
