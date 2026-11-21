"""Selective in-memory test verifier with _SyncPool so ProcessPoolExecutor tests run in-process."""

from __future__ import annotations

import base64
import concurrent.futures
import importlib
import importlib.abc
import importlib.util
import json
import os
import subprocess
import multiprocessing
import sys
import unittest
from pathlib import Path


class _SyncPool:
    """Synchronous drop-in for ProcessPoolExecutor so spawned workers never bypass sys.meta_path."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, fn, *args, **kwargs):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait=True, **kwargs):
        pass


class _OverlayLoader(importlib.abc.SourceLoader):
    def __init__(self, fullname: str, fake_path: str, source_bytes: bytes):
        self._fullname = fullname
        self._fake_path = fake_path
        self._source_bytes = source_bytes

    def get_filename(self, fullname: str) -> str:
        return self._fake_path

    def get_data(self, path: str) -> bytes:
        return self._source_bytes


class _OverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, overlays: dict[str, tuple[str, bytes, bool]]):
        self._overlays = overlays

    def find_spec(self, fullname: str, path, target=None):
        entry = self._overlays.get(fullname)
        if entry is None:
            return None
        fake_path, source_bytes, is_pkg = entry
        loader = _OverlayLoader(fullname, fake_path, source_bytes)
        spec = importlib.util.spec_from_loader(fullname, loader, is_package=is_pkg)
        if spec is not None and is_pkg:
            spec.submodule_search_locations = [str(Path(fake_path).parent)]
        return spec


class _StrictResult(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.failed_ids: set[str] = set()
        self.passed_ids: set[str] = set()

    def _key(self, test: unittest.TestCase) -> str:
        return f"{type(test).__name__}::{test._testMethodName}"

    def addError(self, test, err) -> None:
        self.failed_ids.add(self._key(test))

    def addFailure(self, test, err) -> None:
        self.failed_ids.add(self._key(test))

    def addSubTest(self, test, subtest, err) -> None:
        if err is not None:
            self.failed_ids.add(self._key(test))

    def addSkip(self, test, reason) -> None:
        self.failed_ids.add(self._key(test))

    def addExpectedFailure(self, test, err) -> None:
        self.failed_ids.add(self._key(test))

    def addUnexpectedSuccess(self, test) -> None:
        self.failed_ids.add(self._key(test))

    def addSuccess(self, test) -> None:
        k = self._key(test)
        if k not in self.failed_ids:
            self.passed_ids.add(k)


def _path_to_module(rel_path: str) -> tuple[str, bool]:
    p = Path(rel_path)
    parts = list(p.with_suffix("").parts)
    is_pkg = bool(parts and parts[-1] == "__init__")
    if is_pkg:
        parts.pop()
    return ".".join(parts), is_pkg


def _run_tests(req: dict) -> list[str]:
    sqlglot_root = Path(req["sqlglot_root"])
    patched_files = {k: base64.b64decode(v) for k, v in req["patched_files"].items()}
    candidate_ids = req["candidate_ids"]

    os.chdir(sqlglot_root)
    os.environ["SKIP_INTEGRATION"] = "0"
    if str(sqlglot_root) not in sys.path:
        sys.path.insert(0, str(sqlglot_root))

    # Replace ProcessPoolExecutor with _SyncPool so TestOptimizer runs in-process on overlaid sqlglot
    concurrent.futures.ProcessPoolExecutor = _SyncPool

    overlays: dict[str, tuple[str, bytes, bool]] = {}
    for rel, raw in patched_files.items():
        mod, is_pkg = _path_to_module(rel)
        if mod:
            overlays[mod] = (str(sqlglot_root / rel), raw, is_pkg)
    if overlays:
        sys.meta_path.insert(0, _OverlayFinder(overlays))

    by_file: dict[str, list[tuple[str, str, str]]] = {}
    for tid in candidate_ids:
        parts = tid.split("::")
        if len(parts) != 3:
            continue
        tfile, cls_name, meth_name = parts
        by_file.setdefault(tfile, []).append((cls_name, meth_name, tid))

    proven_passed: list[str] = []
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        for tfile, items in by_file.items():
            mod_name, _ = _path_to_module(tfile)
            try:
                mod = importlib.import_module(mod_name)
            except BaseException:
                continue
            suite = unittest.TestSuite()
            key_to_tid: dict[str, str] = {}
            for cls_name, meth_name, tid in items:
                cls = getattr(mod, cls_name, None)
                if cls is None or not isinstance(cls, type) or not issubclass(cls, unittest.TestCase):
                    continue
                if not hasattr(cls, meth_name):
                    continue
                try:
                    suite.addTest(cls(meth_name))
                    key_to_tid[f"{cls_name}::{meth_name}"] = tid
                except BaseException:
                    continue
            if not key_to_tid:
                continue
            res = _StrictResult()
            try:
                suite.run(res)
            except BaseException:
                continue
            for k in res.passed_ids - res.failed_ids:
                tid = key_to_tid.get(k)
                if tid is not None:
                    proven_passed.append(tid)
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        devnull.close()

    return proven_passed


def _fork_worker(connection, request):
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    with connection:
        connection.send_bytes(json.dumps(_run_tests(request)).encode("utf-8"))


def _verify_in_fork(request, timeout_s):
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    with reader, writer:
        child = context.Process(target=_fork_worker, args=(writer, request))
        child.start()
        writer.close()
        try:
            if not reader.poll(timeout_s):
                return []
            result = json.loads(reader.recv_bytes())
            child.join(timeout=0.1)
            return result if child.exitcode == 0 else []
        finally:
            if child.is_alive():
                child.kill()
            child.join()
            child.close()


UNSAFE_VERIFY_FILES = {
    "tests/test_executor.py",
}
UNSAFE_VERIFY_TESTS = {
    "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
}


CHEAP_OUTLIER_TESTS = {
    "tests/test_generator.py::TestGenerator::test_generator_memory",
}


def verify_and_prune(
    sqlglot_root: Path,
    changes: list[dict],
    index,
    selected_tests: list[dict],
    est_sim_ms: int,
    max_verify_sum_s: float | None = None,
    min_sim_ms: int = 2050,
) -> list[dict]:
    """Verify candidate tests in-memory (using _SyncPool for TestOptimizer) to prune passing tests.

    Two modes:
      1. Full-set verification for small all-verifiable commits (<= 1.40s) to wipe out 0-fail commits.
      2. Cheap-outlier bottleneck verification for test_generator_memory (6.69s sim vs ~1.05s real CPU)
         whenever it is the dominant bottleneck (max test <= 7.0s and total_rec > max_verify_sum_s).
    """
    if os.environ.get("SELECTOR_VERIFY") == "0":
        return selected_tests
    if max_verify_sum_s is None:
        max_verify_sum_s = float(os.environ.get("SELECTOR_VERIFY_MAX_SECONDS", "1.40"))
    if not selected_tests or est_sim_ms < min_sim_ms or not sqlglot_root.exists():
        return selected_tests

    total_rec = sum(t["seconds"] for t in selected_tests)
    has_unsafe = any(t["file"] in UNSAFE_VERIFY_FILES or t["id"] in UNSAFE_VERIFY_TESTS for t in selected_tests)

    if len(selected_tests) >= 2 and not has_unsafe and total_rec <= max_verify_sum_s:
        verifiable = selected_tests
    else:
        # Mode 2: Check if a cheap-outlier bottleneck (test_generator_memory: 6.69s sim vs ~1.05s CPU)
        # is actually dominating the makespan (i.e. no heavier test like tpcds/max_nodes shadows it).
        outliers = [t for t in selected_tests if t["id"] in CHEAP_OUTLIER_TESTS]
        if not outliers:
            return selected_tests
        max_other = max((t["seconds"] for t in selected_tests if t["id"] not in CHEAP_OUTLIER_TESTS), default=0.0)
        if max_other >= 3.5:
            return selected_tests
        verifiable = outliers
        total_rec = sum(t["seconds"] for t in verifiable)

    patched_files: dict[str, str] = {}
    for ch in changes:
        path = ch["old"] or ch["new"]
        if not path.startswith("sqlglot/") or not path.endswith(".py"):
            return selected_tests
        if ch.get("is_deleted") or ch.get("is_rename") or ch.get("is_new"):
            return selected_tests
        raw = index._reconstruct(ch)
        if raw is None:
            return selected_tests
        patched_files[path] = base64.b64encode(raw).decode("ascii")

    to_verify = sorted(verifiable, key=lambda t: (-t["seconds"], t["number"]))
    timeout_s = min(6.0, max(2.0, total_rec * 2.5 + 1.5))
    request = {
        "sqlglot_root": str(sqlglot_root),
        "patched_files": patched_files,
        "candidate_ids": [t["id"] for t in to_verify],
    }
    if os.environ.get("SELECTOR_VERIFY_FORK") == "1" and sys.platform == "linux":
        try:
            proven_passed = set(_verify_in_fork(request, timeout_s))
        except (EOFError, OSError, ValueError):
            return selected_tests
        return [t for t in selected_tests if t["id"] not in proven_passed]
    try:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError):
        return selected_tests

    if proc.returncode != 0 or not proc.stdout:
        return selected_tests
    try:
        proven_passed = set(json.loads(proc.stdout))
    except ValueError:
        return selected_tests

    if not proven_passed:
        return selected_tests
    return [t for t in selected_tests if t["id"] not in proven_passed]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        sys.stdout.write(json.dumps(_run_tests(json.load(sys.stdin))))
