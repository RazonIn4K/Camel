import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-benchmark",
        action="store_true",
        default=False,
        help="run benchmark tests",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-benchmark"):
        skip_benchmark = pytest.mark.skip(reason="need --run-benchmark option to run")
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_benchmark)
