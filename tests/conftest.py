import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# app.py and train.py use paths relative to the repository root.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="also retrain the network")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: retrains the network, run with --run-slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow, run with --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
