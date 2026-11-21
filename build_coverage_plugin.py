"""Record non-source files read while each test is running."""

from __future__ import annotations

import builtins
import io
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from pool_coverage import run_with_context

ROOT = Path(os.environ["SQLGLOT_ROOT"]).resolve()
OUTPUT = Path(f"{os.environ['COVERAGE_ACCESS_FILE']}.{os.environ.get('PYTEST_XDIST_WORKER', 'main')}")
current_test: str | None = None
accesses: dict[str, set[str]] = defaultdict(set)

_open = builtins.open
_io_open = io.open
_submit = ProcessPoolExecutor.submit


def _tracked_submit(pool, function, /, *args, **kwargs):
    if current_test is None:
        return _submit(pool, function, *args, **kwargs)
    return _submit(pool, run_with_context, current_test, function, args, kwargs)


def _record(file: object) -> None:
    if current_test is None or isinstance(file, int):
        return
    try:
        relative = Path(os.fspath(file)).resolve().relative_to(ROOT).as_posix()
    except (OSError, TypeError, ValueError):
        return
    if not relative.endswith(".py"):
        accesses[relative].add(current_test)


def _tracked_open(file: object, *args: object, **kwargs: object):
    _record(file)
    return _open(file, *args, **kwargs)


def _tracked_io_open(file: object, *args: object, **kwargs: object):
    _record(file)
    return _io_open(file, *args, **kwargs)


builtins.open = _tracked_open
io.open = _tracked_io_open
ProcessPoolExecutor.submit = _tracked_submit


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item) -> None:
    global current_test
    current_test = item.nodeid


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem) -> None:
    global current_test
    current_test = None


def pytest_sessionfinish(session, exitstatus) -> None:
    OUTPUT.write_text(
        "".join(
            f"{path}\t{test_id}\n"
            for path, test_ids in sorted(accesses.items())
            for test_id in sorted(test_ids)
        ),
        encoding="utf-8",
    )
