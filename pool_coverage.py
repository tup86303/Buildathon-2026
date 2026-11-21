"""Collect SQLGlot lines executed inside process-pool tasks during the image build."""

import os
from multiprocessing.util import Finalize

from coverage import Coverage

_coverage = None


def _finish():
    _coverage.stop()
    _coverage.save()


def run_with_context(test_id, function, args, kwargs):
    global _coverage
    if _coverage is None:
        _coverage = Coverage(
            data_file=os.environ["COVERAGE_FILE"] + "-pool",
            data_suffix=True,
            source=["sqlglot"],
        )
        _coverage.start()
        Finalize(None, _finish, exitpriority=10)
    _coverage.switch_context(test_id + "|run")
    try:
        return function(*args, **kwargs)
    finally:
        _coverage.switch_context("")
