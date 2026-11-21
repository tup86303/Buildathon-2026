"""Collect SQLGlot lines executed inside process-pool tasks during the image build."""

import os
import pickle
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import Finalize

from coverage import Coverage

_coverage = None


class _EncodedResult:
    def __init__(self, value):
        self.payload = bytes(ForkingPickler.dumps(value))

    def __reduce__(self):
        return pickle.loads, (self.payload,)


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
        return _EncodedResult(function(*args, **kwargs))
    finally:
        _coverage.switch_context("")
