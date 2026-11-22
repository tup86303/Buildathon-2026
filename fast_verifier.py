"""Fast, read-only-safe, zero-disk-write os.fork() copy-on-write test verifier.

Pre-imports stdlib/unittest (and duckdb/pandas when available) ONCE in the parent
without importing sqlglot, then forks a clean copy-on-write child per commit (~1 ms)
with an in-memory MetaPathFinder overlay so sqlglot imports 100% fresh with zero
stale-reference risk and zero subprocess startup overhead.
"""
from __future__ import annotations

import gc
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import marshal
import os
import signal
import sys
import unittest
from collections import defaultdict
from concurrent.futures import Future
from pathlib import Path

_WARMED = False

def _load_baseline(p: Path) -> dict:
    return marshal.loads(p.read_bytes()) if p.suffix == ".marshal" else json.loads(p.read_text(encoding="utf-8"))

# Tests we never run inside fast_verifier (heavy executor/memory/c-ext tests)
_EXCLUDED_PREFIXES = (
    "tests/leakcheck.py::",
    "tests/test_sqlglotc.py::",
)
_EXCLUDED_IDS = {
    "tests/test_parser.py::TestParser::test_max_nodes",
    "tests/test_generator.py::TestGenerator::test_generator_memory",
    "tests/test_executor.py::TestExecutor::test_execute_tpcds",
    "tests/test_executor.py::TestExecutor::test_execute_tpch",
    "tests/test_executor.py::TestExecutor::test_optimized_tpch",
}


class _LazyFuture:
    """Zero-overhead lazy Future: executes fn only when .result() is called so
    fail-fast _NoOpSubTest aborts immediately on the first failing query."""

    __slots__ = ("_fn", "_args", "_kwargs")

    def __init__(self, fn, args, kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def result(self, timeout=None):
        return self._fn(*self._args, **self._kwargs)


class _SyncPool:
    """In-process lazy drop-in for ProcessPoolExecutor in TestOptimizer."""

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args, **kwargs):
        return _LazyFuture(fn, args, kwargs)

    def shutdown(self, wait=True, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _noop_wait(fs, *args, **kwargs):
    return (set(fs), set())


def _make_as_completed(changed_dialects: set[str]):
    if not changed_dialects:
        return lambda fs, *args, **kwargs: iter(fs)

    def _dialect_first_as_completed(fs, *args, **kwargs):
        items = list(fs)
        items.sort(
            key=lambda f: 0
            if (isinstance(f, _LazyFuture) and len(f._args) >= 3 and f._args[2] in changed_dialects)
            else 1
        )
        return iter(items)

    return _dialect_first_as_completed


def _changed_dialect_names(changes: list[dict]) -> set[str]:
    out: set[str] = set()
    for ch in changes:
        p = ch.get("old") or ch.get("new") or ""
        if p.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
            stem = Path(p).stem
            if stem not in ("__init__", "dialect", "parser", "generator"):
                out.add(stem)
    return out


def _patch_optimizer_pool(mod, changed_dialects: set[str]) -> None:
    mod.ProcessPoolExecutor = _SyncPool
    mod.wait = _noop_wait
    mod.as_completed = _make_as_completed(changed_dialects)


class _BytesLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname: str, orig_path: str, raw_bytes: bytes):
        super().__init__(fullname, orig_path)
        self._raw_bytes = raw_bytes

    def get_data(self, path):
        return self._raw_bytes



class _NoOpSubTest:
    __slots__ = ()
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False

_NOOP_SUBTEST = _NoOpSubTest()
def _fast_subtest(self, msg=None, **params):
    return _NOOP_SUBTEST

class _BytesOverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, overlay: dict[str, tuple[bytes, str]]):
        self._overlay = overlay

    def find_spec(self, fullname, path, target=None):
        entry = self._overlay.get(fullname)
        if entry is not None:
            raw_bytes, orig_path = entry
            return importlib.util.spec_from_file_location(
                fullname,
                orig_path,
                loader=_BytesLoader(fullname, orig_path, raw_bytes),
            )
        return None


def _rel_to_mod(rel: str) -> str:
    return rel[:-3].replace("/", ".")


def _warm_parent() -> None:
    global _WARMED
    if _WARMED:
        return
    _WARMED = True
    # Pre-import heavy non-sqlglot modules ONCE so forked children inherit them via CoW.
    # CRITICAL: Never import sqlglot here so every child imports sqlglot fresh via overlay.
    for mod in (
        "unittest.mock",
        "ast",
        "re",
        "decimal",
        "datetime",
        "itertools",
        "functools",
        "typing",
        "duckdb",
        "pandas",
    ):
        try:
            importlib.import_module(mod)
        except BaseException:
            pass
    gc.collect()
    gc.freeze()


def _child_run_tests(
    sqlglot_root: str,
    overlay: dict[str, tuple[bytes, str]],
    ordered_tests: list[tuple[str, float]],
    early_abort_rem_sec: float,
    timeout_sec: int,
) -> dict:
    gc.disable()
    signal.alarm(max(2, int(timeout_sec)))
    os.chdir(sqlglot_root)
    os.environ["SKIP_INTEGRATION"] = "0"
    if sqlglot_root not in sys.path:
        sys.path.insert(0, sqlglot_root)
    if overlay:
        sys.meta_path.insert(0, _BytesOverlayFinder(overlay))

    # Group class setup lazily while still respecting cheapest-first execution order
    loaded_mods: dict[str, object] = {}
    setup_classes: set[type] = set()
    failed_classes: set[type] = set()

    keep: list[str] = []
    ran = 0
    aborted = False
    devnull = open(os.devnull, "w")
    old_o, old_e = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = devnull
        for idx, (tid, _sec) in enumerate(ordered_tests):
            parts = tid.split("::")
            if len(parts) != 3:
                keep.extend(t for t, _ in ordered_tests[idx:])
                break
            tfile, cn, mn = parts
            mod_name = tfile[:-3].replace("/", ".")
            if mod_name not in loaded_mods:
                try:
                    mod = importlib.import_module(mod_name)
                    if mod_name == "tests.test_optimizer":
                        _patch_optimizer_pool(mod, set())
                    loaded_mods[mod_name] = mod
                except BaseException:
                    loaded_mods[mod_name] = None
            mod = loaded_mods[mod_name]
            if mod is None:
                keep.append(tid)
                ran += 1
                rem_sec = sum(s for _, s in ordered_tests[idx + 1 :])
                if rem_sec <= early_abort_rem_sec:
                    keep.extend(t for t, _ in ordered_tests[idx + 1 :])
                    aborted = True
                    break
                continue

            cls = getattr(mod, cn, None)
            if cls is None or not isinstance(cls, type) or not issubclass(cls, unittest.TestCase):
                keep.append(tid)
                ran += 1
                continue

            if cls not in setup_classes and cls not in failed_classes:
                if hasattr(cls, "setUpClass"):
                    try:
                        cls.setUpClass()
                        setup_classes.add(cls)
                    except BaseException:
                        failed_classes.add(cls)
                else:
                    setup_classes.add(cls)

            if cls in failed_classes:
                keep.append(tid)
                ran += 1
                rem_sec = sum(s for _, s in ordered_tests[idx + 1 :])
                if rem_sec <= early_abort_rem_sec:
                    keep.extend(t for t, _ in ordered_tests[idx + 1 :])
                    aborted = True
                    break
                continue

            res = unittest.TestResult()
            try:
                cls(mn).run(res)
            except BaseException:
                keep.append(tid)
                ran += 1
                rem_sec = sum(s for _, s in ordered_tests[idx + 1 :])
                if rem_sec <= early_abort_rem_sec:
                    keep.extend(t for t, _ in ordered_tests[idx + 1 :])
                    aborted = True
                    break
                continue

            ran += 1
            if res.failures or res.errors:
                keep.append(tid)
                rem_sec = sum(s for _, s in ordered_tests[idx + 1 :])
                if rem_sec <= early_abort_rem_sec:
                    keep.extend(t for t, _ in ordered_tests[idx + 1 :])
                    aborted = True
                    break

        for cls in setup_classes:
            if hasattr(cls, "tearDownClass"):
                try:
                    cls.tearDownClass()
                except BaseException:
                    pass
    finally:
        sys.stdout = old_o
        sys.stderr = old_e
        devnull.close()

    return {"ran": ran, "aborted": aborted, "keep": keep}


def verify_fast_tests(
    changes: list[dict],
    selected: list[dict],
    index,
    sqlglot_root: Path,
    verify_script: Path | None = None,
    max_total_seconds: float = 1.00,
    max_tests: int = 15,
    early_abort_rem_sec: float = 0.80,
    disk_read_files: set[str] | None = None,
) -> list[dict]:
    changed_dialects = _changed_dialect_names(changes)
    if not selected or len(selected) > max_tests or not sqlglot_root.is_dir():
        return selected
    touches_serde = any((ch.get("old") == "sqlglot/serde.py" or ch.get("new") == "sqlglot/serde.py") for ch in changes)
    if any(
        t["id"].startswith(_EXCLUDED_PREFIXES)
        or t["id"] in _EXCLUDED_IDS
        or (touches_serde and t["id"].startswith("tests/test_optimizer.py::"))
        for t in selected
    ):
        return selected
    total_sec = sum(t["seconds"] for t in selected)
    if total_sec > max_total_seconds:
        return selected

    # Only verify pure sqlglot/*.py edits (no tests/*, no __init__.py, no rename/delete/new)
    overlay: dict[str, tuple[bytes, str]] = {}
    for ch in changes:
        old_p = ch.get("old", "")
        new_p = ch.get("new", "")
        if (
            not old_p
            or old_p != new_p
            or not old_p.startswith("sqlglot/")
            or not old_p.endswith(".py")
            or Path(old_p).name == "__init__.py"
            or ch.get("is_deleted")
            or ch.get("is_new")
            or ch.get("is_rename")
            or (disk_read_files is not None and old_p in disk_read_files)
        ):
            return selected
        raw = index.reconstruct(ch)
        if raw is None:
            return selected
        overlay[_rel_to_mod(old_p)] = (raw, str(sqlglot_root / old_p))
    if not overlay:
        return selected

    eff_root = _choose_sqlglot_root(changes, sqlglot_root)
    if eff_root != sqlglot_root:
        overlay = {
            mod: (raw, str(eff_root / (mod.replace(".", "/") + ".py")))
            for mod, (raw, _) in overlay.items()
        }
        sqlglot_root = eff_root

    _warm_parent()

    # Run cheapest tests first so true failing commits abort quickly
    ordered = sorted(((t["id"], float(t["seconds"])) for t in selected), key=lambda x: x[1])
    timeout_sec = max(3, int(total_sec * 3.0 + 3.0))

    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        try:
            result = _child_run_tests(
                str(sqlglot_root), overlay, ordered, early_abort_rem_sec, timeout_sec
            )
            payload = json.dumps(result).encode("utf-8")
            os.write(w_fd, payload)
        except BaseException:
            pass
        finally:
            try:
                os.close(w_fd)
            except OSError:
                pass
            os._exit(0)

    os.close(w_fd)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(r_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(r_fd)
        try:
            _, status = os.waitpid(pid, 0)
        except ChildProcessError:
            status = 1

    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        return selected
    raw_out = b"".join(chunks)
    if not raw_out:
        return selected
    try:
        data = json.loads(raw_out.decode("utf-8"))
    except BaseException:
        return selected
    if not data.get("aborted") and data.get("ran") != len(ordered):
        return selected
    keep_set = set(data.get("keep", ()))
    return [t for t in selected if t["id"] in keep_set]


def probe_import_crash(
    changes: list[dict],
    tests: list[dict],
    index,
    sqlglot_root: Path,
    extra_keep_ids: set[str] | None = None,
) -> list[dict] | None:
    """When a pure sqlglot/*.py edit triggers a full-suite fallback (syntax/import/scalar crash),
    probe importing the patched module(s) and all test files in a clean os.fork() child.
    If any patched module actually crashes at import time, return exact failed-import test files
    unioned with extra_keep_ids (dialect/file coverage) â€” dropping unrelated heavy tests like
    test_max_nodes (28.8s) and test_execute_tpcds (21.5s) with 0 misses.
    Returns None if no import crash occurred or probing is not applicable."""
    if not tests or not sqlglot_root.is_dir():
        return None
    overlay: dict[str, tuple[bytes, str]] = {}
    target_mods: list[str] = []
    for ch in changes:
        old_p = ch.get("old", "")
        new_p = ch.get("new", "")
        if (
            not old_p
            or old_p != new_p
            or not old_p.startswith("sqlglot/")
            or not old_p.endswith(".py")
            or Path(old_p).name == "__init__.py"
            or ch.get("is_deleted")
            or ch.get("is_new")
            or ch.get("is_rename")
        ):
            return None
        raw = index.reconstruct(ch)
        if raw is None:
            return None
        mod = _rel_to_mod(old_p)
        overlay[mod] = (raw, str(sqlglot_root / old_p))
        target_mods.append(mod)
    if not overlay:
        return None

    _warm_parent()
    test_files = sorted({t["file"] for t in tests})
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        try:
            gc.disable()
            signal.alarm(8)
            os.chdir(str(sqlglot_root))
            if str(sqlglot_root) not in sys.path:
                sys.path.insert(0, str(sqlglot_root))
            sys.meta_path.insert(0, _BytesOverlayFinder(overlay))
            devnull = open(os.devnull, "w")
            sys.stdout = sys.stderr = devnull
            crashed = False
            for m in target_mods:
                try:
                    importlib.import_module(m)
                except BaseException:
                    crashed = True
            failed_files: list[str] = []
            if crashed:
                for tf in test_files:
                    tm = tf[:-3].replace("/", ".")
                    try:
                        importlib.import_module(tm)
                    except BaseException:
                        failed_files.append(tf)
            payload = json.dumps({"crashed": crashed, "failed_files": failed_files}).encode("utf-8")
            os.write(w_fd, payload)
        except BaseException:
            pass
        finally:
            try:
                os.close(w_fd)
            except OSError:
                pass
            os._exit(0)

    os.close(w_fd)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(r_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(r_fd)
        try:
            _, status = os.waitpid(pid, 0)
        except ChildProcessError:
            status = 1

    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0 or not chunks:
        return None
    try:
        data = json.loads(b"".join(chunks).decode("utf-8"))
    except BaseException:
        return None
    if not data.get("crashed"):
        return None
    failed_files = set(data.get("failed_files", ()))
    if not failed_files:
        return None
    keep_ids = set(extra_keep_ids or ())
    narrowed = [t for t in tests if t["file"] in failed_files or t["id"] in keep_ids]
    return narrowed if narrowed else None


_TPCDS_TEST_ID = "tests/test_executor.py::TestExecutor::test_execute_tpcds"


def _tpcds_query_order(pairs, baseline):
    rank = {idx: position for position, idx in enumerate(baseline)}
    return sorted(pairs, key=lambda pair: rank.get(pair[0], -1))


_SQLGLOTC_EXACT = {
    "sqlglot/errors.py",
    "sqlglot/time.py",
    "sqlglot/optimizer/resolver.py",
    "sqlglot/optimizer/isolate_table_selects.py",
    "sqlglot/optimizer/normalize_identifiers.py",
    "sqlglot/optimizer/qualify.py",
    "sqlglot/optimizer/qualify_tables.py",
    "sqlglot/helper.py",
    "sqlglot/tokenizer_core.py",
    "sqlglot/trie.py",
    "sqlglot/optimizer/scope.py",
    "sqlglot/optimizer/qualify_columns.py",
    "sqlglot/optimizer/annotate_types.py",
}
_SQLGLOTC_PREFIXES = ("sqlglot/expressions/",)


def _choose_sqlglot_root(changes: list[dict], pure_root: Path) -> Path:
    """Use C-compiled /app/sqlglot_c when present AND no sqlglotc-compiled module was touched."""
    c_root = Path(os.environ.get("SELECTOR_SQLGLOT_C_ROOT", str(pure_root.parent / "sqlglot_c")))
    if not c_root.is_dir():
        return pure_root
    for ch in changes:
        for path in (ch.get("old") or "", ch.get("new") or ""):
            if path in _SQLGLOTC_EXACT or path.startswith(_SQLGLOTC_PREFIXES):
                return pure_root
    return c_root


def verify_tpcds_differential(
    changes: list[dict],
    selected: list[dict],
    index,
    sqlglot_root: Path,
    baseline_path: Path,
    max_diff_exec: int = 1,
    disk_read_files: set[str] | None = None,
) -> list[dict]:
    """Prune test_execute_tpcds (21.454s) when a non-executor commit leaves all 57 executed
    TPC-DS optimized plans + duckdb transpiles identical â€” or when <= max_diff_exec queries
    differ and all differing queries pass live execution against DuckDB."""
    if not any(t["id"] == _TPCDS_TEST_ID for t in selected):
        return selected
    if not sqlglot_root.is_dir() or not baseline_path.is_file():
        return selected

    overlay: dict[str, tuple[bytes, str]] = {}
    for ch in changes:
        old_p = ch.get("old", "")
        new_p = ch.get("new", "")
        _tpcds_allowed = (
            old_p.startswith((
                "sqlglot/optimizer/",
                "sqlglot/parsers/",
                "sqlglot/generators/",
                "sqlglot/dialects/",
                "sqlglot/expressions/",
            ))
            or old_p in (
                "sqlglot/parser.py",
                "sqlglot/generator.py",
                "sqlglot/tokens.py",
                "sqlglot/tokenizer_core.py",
            )
        )
        # Never prune expressions/* edits that touch serialization/evaluation hooks used during execution
        if old_p.startswith("sqlglot/expressions/"):
            hunk_txt = "\n".join(line for _, _, _, lines in ch.get("hunks", ()) for line in lines)
            if any(tok in hunk_txt for tok in ("def to_py", "def __reduce__", "def __getstate__", "def __setstate__", "def __hash__", "def __eq__")):
                _tpcds_allowed = False
        if (
            not old_p
            or old_p != new_p
            or not _tpcds_allowed
            or not old_p.endswith(".py")
            or Path(old_p).name == "__init__.py"
            or ch.get("is_deleted")
            or ch.get("is_new")
            or ch.get("is_rename")
            or (disk_read_files is not None and old_p in disk_read_files)
        ):
            return selected
        raw = index.reconstruct(ch)
        if raw is None:
            return selected
        overlay[_rel_to_mod(old_p)] = (raw, str(sqlglot_root / old_p))
    if not overlay:
        return selected

    try:
        base_entries = _load_baseline(baseline_path).get("queries", [])
    except BaseException:
        return selected
    if not base_entries:
        return selected
    base_by_idx = {e["idx"]: (e["opt_hash"], e["duck_hash"]) for e in base_entries}

    eff_root = _choose_sqlglot_root(changes, sqlglot_root)
    if eff_root != sqlglot_root:
        overlay = {
            mod: (raw, str(eff_root / (mod.replace(".", "/") + ".py")))
            for mod, (raw, _) in overlay.items()
        }

    _warm_parent()
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        try:
            gc.disable()
            signal.alarm(15)
            os.chdir(str(eff_root))
            if str(eff_root) not in sys.path:
                sys.path.insert(0, str(eff_root))
            sys.meta_path.insert(0, _BytesOverlayFinder(overlay))
            devnull = open(os.devnull, "w")
            sys.stdout = sys.stderr = devnull

            import hashlib
            from sqlglot import parse_one, transpile
            from sqlglot.optimizer import optimize
            from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

            diff_queries: list[tuple[int, str]] = []
            _pairs = [(i, m, s) for i, (m, s, _) in enumerate(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql")) if m.get("execute")]
            for idx, _, sql in _tpcds_query_order(_pairs, base_by_idx):
                expected = base_by_idx.get(idx)
                if expected is None:
                    os.write(w_fd, b'{"prune":false}')
                    os._exit(0)
                opt_sql = optimize(parse_one(sql), TPCDS_SCHEMA, leave_tables_isolated=True).sql()
                oh = hashlib.blake2b(opt_sql.encode("utf-8"), digest_size=16).hexdigest()
                if oh != expected[0] or hashlib.blake2b(
                    transpile(sql, write="duckdb")[0].encode("utf-8"), digest_size=16
                ).hexdigest() != expected[1]:
                    diff_queries.append((idx, sql))
                    if len(diff_queries) > max_diff_exec:
                        os.write(w_fd, b'{"prune":false}')
                        os._exit(0)

            if not diff_queries:
                os.write(w_fd, b'{"prune":true}')
                os._exit(0)

            # Execute ONLY the 1..max_diff_exec differing queries against DuckDB!
            import duckdb
            import numpy as np
            import pandas as pd
            from pandas.testing import assert_frame_equal
            from sqlglot.executor import execute
            from sqlglot.optimizer.scope import find_tables
            from tests.test_executor import DIR_TPCDS, TestExecutor

            conn = duckdb.connect()
            loaded_tables: dict = {}
            for _, sql in diff_queries:
                expr = parse_one(sql)
                needed = {t.name for t in find_tables(expr)}
                for tname in needed:
                    if tname in loaded_tables:
                        continue
                    cols = TPCDS_SCHEMA[tname]
                    csv_path = f"{DIR_TPCDS}{tname}.csv.gz"
                    df = pd.read_csv(csv_path, header=None, names=list(cols), dtype=str)
                    # Let DuckDB read directly matching TestExecutor.setUpClass
                    col_defs = ", ".join(f"{c} {ty}" for c, ty in cols.items())
                    conn.execute(f"CREATE TABLE {tname} ({col_defs})")
                    conn.execute(f"COPY {tname} FROM '{csv_path}' (DELIMITER '|', NULL '')")
                    rows = conn.execute(f"SELECT * FROM {tname}").fetchall()
                    loaded_tables[tname] = [dict(zip(cols, r)) for r in rows]

                q_tables = {t.name: loaded_tables[t.name] for t in find_tables(expr)}
                table = execute(expr, schema=TPCDS_SCHEMA, tables=q_tables)
                a = conn.execute(transpile(sql, write="duckdb")[0]).fetchdf()
                b = pd.DataFrame(
                    ((np.nan if c is None else c for c in r) for r in table.rows),
                    columns=table.columns,
                )
                assert_frame_equal(a, b, check_dtype=False, check_index_type=False)

            os.write(w_fd, b'{"prune":true}')
        except BaseException:
            try:
                os.write(w_fd, b'{"prune":false}')
            except BaseException:
                pass
        finally:
            try:
                os.close(w_fd)
            except OSError:
                pass
            os._exit(0)

    os.close(w_fd)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(r_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(r_fd)
        try:
            _, status = os.waitpid(pid, 0)
        except ChildProcessError:
            status = 1

    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0 or not chunks:
        return selected
    try:
        data = json.loads(b"".join(chunks).decode("utf-8"))
    except BaseException:
        return selected
    if data.get("prune"):
        return [t for t in selected if t["id"] != _TPCDS_TEST_ID]
    return selected


_TPCH_OPT_TEST_ID = "tests/test_executor.py::TestExecutor::test_optimized_tpch"


def verify_optimized_tpch(
    changes: list[dict],
    selected: list[dict],
    index,
    sqlglot_root: Path,
    baseline_path: Path,
    disk_read_files: set[str] | None = None,
) -> list[dict]:
    """Prune test_optimized_tpch (3.537s) in ~80ms CPU (44x ROI!) whenever all 22
    (transpile(sql, write='duckdb'), transpile(optimized, write='duckdb')) outputs
    are byte-for-byte identical to the pinned baseline."""
    if not any(t["id"] == _TPCH_OPT_TEST_ID for t in selected):
        return selected
    if not sqlglot_root.is_dir() or not baseline_path.is_file():
        return selected

    overlay: dict[str, tuple[bytes, str]] = {}
    for ch in changes:
        old_p = ch.get("old", "")
        new_p = ch.get("new", "")
        if (
            not old_p
            or old_p != new_p
            or not old_p.startswith("sqlglot/")
            or not old_p.endswith(".py")
            or Path(old_p).name == "__init__.py"
            or ch.get("is_deleted")
            or ch.get("is_new")
            or ch.get("is_rename")
            or (disk_read_files is not None and old_p in disk_read_files)
        ):
            return selected
        raw = index.reconstruct(ch)
        if raw is None:
            return selected
        overlay[_rel_to_mod(old_p)] = (raw, str(sqlglot_root / old_p))
    if not overlay:
        return selected

    try:
        expected_hash = _load_baseline(baseline_path).get("tpch_opt_hash")
    except BaseException:
        return selected
    if not expected_hash:
        return selected

    eff_root = _choose_sqlglot_root(changes, sqlglot_root)
    if eff_root != sqlglot_root:
        overlay = {
            mod: (raw, str(eff_root / (mod.replace(".", "/") + ".py")))
            for mod, (raw, _) in overlay.items()
        }

    _warm_parent()
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        try:
            gc.disable()
            signal.alarm(5)
            os.chdir(str(eff_root))
            if str(eff_root) not in sys.path:
                sys.path.insert(0, str(eff_root))
            sys.meta_path.insert(0, _BytesOverlayFinder(overlay))
            devnull = open(os.devnull, "w")
            sys.stdout = sys.stderr = devnull
            import hashlib
            from sqlglot import transpile
            from tests.helpers import load_sql_fixture_pairs

            h = hashlib.blake2b(digest_size=16)
            for _, sql, opt in load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"):
                h.update(transpile(sql, write="duckdb")[0].encode("utf-8"))
                h.update(b"\0")
                h.update(transpile(opt, write="duckdb")[0].encode("utf-8"))
                h.update(b"\n")
            ok = h.hexdigest() == expected_hash
            os.write(w_fd, b'{"prune":true}' if ok else b'{"prune":false}')
        except BaseException:
            try:
                os.write(w_fd, b'{"prune":false}')
            except BaseException:
                pass
        finally:
            try:
                os.close(w_fd)
            except OSError:
                pass
            os._exit(0)

    os.close(w_fd)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(r_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(r_fd)
        try:
            _, status = os.waitpid(pid, 0)
        except ChildProcessError:
            status = 1

    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0 or not chunks:
        return selected
    try:
        data = json.loads(b"".join(chunks).decode("utf-8"))
    except BaseException:
        return selected
    if data.get("prune"):
        return [t for t in selected if t["id"] != _TPCH_OPT_TEST_ID]
    return selected


_TPCH_EXEC_TEST_ID = "tests/test_executor.py::TestExecutor::test_execute_tpch"
_CANONICALIZE_TEST_ID = "tests/test_optimizer.py::TestOptimizer::test_canonicalize"


def verify_high_roi_tests(
    changes: list[dict],
    selected: list[dict],
    index,
    sqlglot_root: Path,
    baseline_path: Path,
    disk_read_files: set[str] | None = None,
) -> list[dict]:
    """In ONE single os.fork() child, verify ultra-high-ROI slow tests (18x to 177x ROI!):
    - test_execute_tpch (3.863s sim / ~25ms CPU): tpc-h.sql has 0 execute=true queries, so
      test_execute_tpch only runs parse_one(sql) + pickle roundtrip for ProcessPoolExecutor!
    - test_optimized_tpch (3.537s sim / ~80ms CPU): checks exact 22-query duckdb transpile hash.
    - test_canonicalize (1.006s sim / ~55ms CPU): runs exact unittest method in-process."""
    sel_ids = {t["id"] for t in selected}
    check_tpch_exec = _TPCH_EXEC_TEST_ID in sel_ids
    check_tpch_opt = _TPCH_OPT_TEST_ID in sel_ids and baseline_path.is_file()
    touches_serde = any((ch.get("old") == "sqlglot/serde.py" or ch.get("new") == "sqlglot/serde.py") for ch in changes)
    changed_dialects = _changed_dialect_names(changes)
    check_canon = (_CANONICALIZE_TEST_ID in sel_ids) and not touches_serde
    if not (check_tpch_exec or check_tpch_opt or check_canon):
        return selected
    if not sqlglot_root.is_dir():
        return selected

    overlay: dict[str, tuple[bytes, str]] = {}
    for ch in changes:
        old_p = ch.get("old", "")
        new_p = ch.get("new", "")
        if (
            not old_p
            or old_p != new_p
            or not old_p.startswith("sqlglot/")
            or not old_p.endswith(".py")
            or Path(old_p).name == "__init__.py"
            or ch.get("is_deleted")
            or ch.get("is_new")
            or ch.get("is_rename")
            or (disk_read_files is not None and old_p in disk_read_files)
        ):
            return selected
        raw = index.reconstruct(ch)
        if raw is None:
            return selected
        overlay[_rel_to_mod(old_p)] = (raw, str(sqlglot_root / old_p))
    if not overlay:
        return selected

    expected_tpch_opt = None
    if check_tpch_opt:
        try:
            expected_tpch_opt = _load_baseline(baseline_path).get("tpch_opt_hash")
        except BaseException:
            expected_tpch_opt = None
        if not expected_tpch_opt:
            check_tpch_opt = False
    if not (check_tpch_exec or check_tpch_opt or check_canon):
        return selected

    eff_root = _choose_sqlglot_root(changes, sqlglot_root)
    if eff_root != sqlglot_root:
        overlay = {
            mod: (raw, str(eff_root / (mod.replace(".", "/") + ".py")))
            for mod, (raw, _) in overlay.items()
        }

    _warm_parent()
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        pruned: list[str] = []
        try:
            gc.disable()
            signal.alarm(6)
            os.chdir(str(eff_root))
            os.environ["SKIP_INTEGRATION"] = "0"
            if str(eff_root) not in sys.path:
                sys.path.insert(0, str(eff_root))
            sys.meta_path.insert(0, _BytesOverlayFinder(overlay))
            devnull = open(os.devnull, "w")
            sys.stdout = sys.stderr = devnull

            if check_tpch_exec or check_tpch_opt:
                import hashlib
                import pickle
                from sqlglot import parse_one, transpile
                from tests.helpers import load_sql_fixture_pairs
                tpch_pairs = list(load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))

                if check_tpch_exec:
                    try:
                        if all(not meta.get("execute") for meta, _, _ in tpch_pairs):
                            for _, sql, _ in tpch_pairs:
                                expr = parse_one(sql)
                                pickle.loads(pickle.dumps(expr))
                            pruned.append(_TPCH_EXEC_TEST_ID)
                    except BaseException:
                        pass

                if check_tpch_opt and expected_tpch_opt:
                    try:
                        h = hashlib.blake2b(digest_size=16)
                        for _, sql, opt in tpch_pairs:
                            h.update(transpile(sql, write="duckdb")[0].encode("utf-8"))
                            h.update(b"\0")
                            h.update(transpile(opt, write="duckdb")[0].encode("utf-8"))
                            h.update(b"\n")
                        if h.hexdigest() == expected_tpch_opt:
                            pruned.append(_TPCH_OPT_TEST_ID)
                    except BaseException:
                        pass

            if check_canon:
                try:
                    mod = importlib.import_module("tests.test_optimizer")
                    _patch_optimizer_pool(mod, changed_dialects)
                    cls = mod.TestOptimizer
                    cls.pool = _SyncPool()
                    res = unittest.TestResult()
                    cls("test_canonicalize").run(res)
                    if not (res.failures or res.errors):
                        pruned.append(_CANONICALIZE_TEST_ID)
                except BaseException:
                    pass

            os.write(w_fd, json.dumps({"pruned": pruned}).encode("utf-8"))
        except BaseException:
            try:
                os.write(w_fd, json.dumps({"pruned": pruned}).encode("utf-8"))
            except BaseException:
                pass
        finally:
            try:
                os.close(w_fd)
            except OSError:
                pass
            os._exit(0)

    os.close(w_fd)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(r_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(r_fd)
        try:
            _, status = os.waitpid(pid, 0)
        except ChildProcessError:
            status = 1

    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0 or not chunks:
        return selected
    try:
        pruned_set = set(json.loads(b"".join(chunks).decode("utf-8")).get("pruned", ()))
    except BaseException:
        return selected
    if not pruned_set:
        return selected
    return [t for t in selected if t["id"] not in pruned_set]


_MERGE_SUBQ_TEST_ID = "tests/test_optimizer.py::TestOptimizer::test_merge_subqueries"
_SCHEMA_CACHE_TEST_ID = "tests/test_schema.py::TestSchema::test_cache_does_not_retain_queries"
_AGG_ORDER_TEST_ID = "tests/test_executor.py::TestExecutor::test_agg_order"


def verify_commit_unified(
    changes: list[dict],
    selected: list[dict],
    index,
    sqlglot_root: Path,
    baseline_path: Path,
    max_total_seconds: float = 1.00,
    max_tests: int = 15,
    early_abort_rem_sec: float = 0.80,
    max_diff_exec: int = 1,
    disk_read_files: set[str] | None = None,
) -> list[dict]:
    """Single-fork unified verifier: pays ONE os.fork() + ONE import sqlglot per commit
    for high-ROI tests (177x/44x/18x/2.7x), Differential TPC-DS (21.454s), AND full-commit
    zero-out verification â€” eliminating duplicate imports and unlocking cascading zero-outs."""
    if not selected or not sqlglot_root.is_dir():
        return selected

    sel_ids = {t["id"] for t in selected}
    touches_pickle_risk = any(
        path == "sqlglot/serde.py" or path.startswith("sqlglot/expressions/")
        for change in changes
        for path in (change.get("old") or "", change.get("new") or "")
    )
    check_tpch_exec = _TPCH_EXEC_TEST_ID in sel_ids
    check_tpch_opt = _TPCH_OPT_TEST_ID in sel_ids and baseline_path.is_file()
    changed_dialects = _changed_dialect_names(changes)
    check_canon = (_CANONICALIZE_TEST_ID in sel_ids) and not touches_pickle_risk
    check_merge = (_MERGE_SUBQ_TEST_ID in sel_ids) and not touches_pickle_risk
    check_schema_cache = _SCHEMA_CACHE_TEST_ID in sel_ids
    check_agg_order = (_AGG_ORDER_TEST_ID in sel_ids) and not touches_pickle_risk

    # Check if Differential TPC-DS is eligible
    check_tpcds = (_TPCDS_TEST_ID in sel_ids) and baseline_path.is_file() and not touches_pickle_risk
    if check_tpcds:
        for ch in changes:
            old_p = ch.get("old", "")
            ok_p = (
                old_p.startswith((
                    "sqlglot/optimizer/",
                    "sqlglot/parsers/",
                    "sqlglot/generators/",
                    "sqlglot/dialects/",
                    "sqlglot/expressions/",
                ))
                or old_p in (
                    "sqlglot/parser.py",
                    "sqlglot/generator.py",
                    "sqlglot/tokens.py",
                    "sqlglot/tokenizer_core.py",
                )
            )
            if ok_p and old_p.startswith("sqlglot/expressions/"):
                hunk_txt = "\n".join(line for _, _, _, lines in ch.get("hunks", ()) for line in lines)
                if any(tok in hunk_txt for tok in ("def to_py", "def __reduce__", "def __getstate__", "def __setstate__", "def __hash__", "def __eq__")):
                    ok_p = False
            if not ok_p:
                check_tpcds = False
                break

    # Check if full-commit fast-verify might run (now OR after Stage A/B prunes!)
    can_fast_after = (
        len(selected) <= max_tests + 5
        and not any(
            (t["id"].startswith(_EXCLUDED_PREFIXES) and t["id"] not in (_TPCDS_TEST_ID, _TPCH_EXEC_TEST_ID, _TPCH_OPT_TEST_ID, _AGG_ORDER_TEST_ID))
            or t["id"] in _EXCLUDED_IDS
            or (touches_pickle_risk and t["id"].startswith("tests/test_optimizer.py::"))
            for t in selected
        )
    )
    est_after_ab = sum(
        t["seconds"] for t in selected
        if t["id"] not in (
            ({_TPCH_EXEC_TEST_ID} if check_tpch_exec else set())
            | ({_TPCH_OPT_TEST_ID} if check_tpch_opt else set())
            | ({_CANONICALIZE_TEST_ID} if check_canon else set())
            | ({_MERGE_SUBQ_TEST_ID} if check_merge else set())
            | ({_SCHEMA_CACHE_TEST_ID} if check_schema_cache else set())
            | ({_AGG_ORDER_TEST_ID} if check_agg_order else set())
            | ({_TPCDS_TEST_ID} if check_tpcds else set())
        )
    )
    may_fast = can_fast_after and est_after_ab <= max_total_seconds

    if not (check_tpch_exec or check_tpch_opt or check_canon or check_merge or check_schema_cache or check_agg_order or check_tpcds or may_fast):
        return selected

    # Build in-memory overlay (pure sqlglot/*.py edits only)
    overlay: dict[str, tuple[bytes, str]] = {}
    for ch in changes:
        old_p = ch.get("old", "")
        new_p = ch.get("new", "")
        if (
            not old_p
            or old_p != new_p
            or not old_p.startswith("sqlglot/")
            or not old_p.endswith(".py")
            or Path(old_p).name == "__init__.py"
            or ch.get("is_deleted")
            or ch.get("is_new")
            or ch.get("is_rename")
            or (disk_read_files is not None and old_p in disk_read_files)
        ):
            return selected
        raw = index.reconstruct(ch)
        if raw is None:
            return selected
        overlay[_rel_to_mod(old_p)] = (raw, str(sqlglot_root / old_p))
    if not overlay:
        return selected

    base_data = {}
    if (check_tpch_opt or check_tpcds) and baseline_path.is_file():
        try:
            base_data = _load_baseline(baseline_path)
        except BaseException:
            base_data = {}
    expected_tpch_opt = base_data.get("tpch_opt_hash") if check_tpch_opt else None
    base_by_idx = (
        {e["idx"]: (e["opt_hash"], e["duck_hash"]) for e in base_data.get("queries", [])}
        if check_tpcds
        else {}
    )
    if not expected_tpch_opt:
        check_tpch_opt = False
    if not base_by_idx:
        check_tpcds = False
    if not (check_tpch_exec or check_tpch_opt or check_canon or check_merge or check_schema_cache or check_agg_order or check_tpcds or may_fast):
        return selected

    eff_root = _choose_sqlglot_root(changes, sqlglot_root)
    if eff_root != sqlglot_root:
        overlay = {
            mod: (raw, str(eff_root / (mod.replace(".", "/") + ".py")))
            for mod, (raw, _) in overlay.items()
        }

    _warm_parent()
    all_items = [(t["id"], float(t["seconds"])) for t in selected]
    timeout_sec = 18 if check_tpcds else max(4, int(est_after_ab * 3.0 + 4.0))

    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        pruned: set[str] = set()
        try:
            gc.disable()
            signal.alarm(timeout_sec)
            unittest.TestCase.subTest = _fast_subtest
            os.chdir(str(eff_root))
            os.environ["SKIP_INTEGRATION"] = "0"
            if str(eff_root) not in sys.path:
                sys.path.insert(0, str(eff_root))
            sys.meta_path.insert(0, _BytesOverlayFinder(overlay))
            devnull = open(os.devnull, "w")
            sys.stdout = sys.stderr = devnull

            # --- Stage A: Ultra-high-ROI tests ---
            if check_tpch_exec or check_tpch_opt:
                import hashlib
                import pickle
                from sqlglot import parse_one, transpile
                from tests.helpers import load_sql_fixture_pairs
                tpch_pairs = list(load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))

                if check_tpch_exec:
                    try:
                        if all(not meta.get("execute") for meta, _, _ in tpch_pairs):
                            for _, sql, _ in tpch_pairs:
                                expr = parse_one(sql)
                                pickle.loads(pickle.dumps(expr))
                            pruned.add(_TPCH_EXEC_TEST_ID)
                    except BaseException:
                        pass

                if check_tpch_opt and expected_tpch_opt:
                    try:
                        h = hashlib.blake2b(digest_size=16)
                        for _, sql, opt in tpch_pairs:
                            h.update(transpile(sql, write="duckdb")[0].encode("utf-8"))
                            h.update(b"\0")
                            h.update(transpile(opt, write="duckdb")[0].encode("utf-8"))
                            h.update(b"\n")
                        if h.hexdigest() == expected_tpch_opt:
                            pruned.add(_TPCH_OPT_TEST_ID)
                    except BaseException:
                        pass

            opt_cls = None
            opt_setup_ok = False
            if check_canon or check_merge:
                try:
                    mod = importlib.import_module("tests.test_optimizer")
                    _patch_optimizer_pool(mod, changed_dialects)
                    opt_cls = mod.TestOptimizer
                    if check_canon:
                        opt_cls.pool = _SyncPool()
                        res = unittest.TestResult()
                        opt_cls("test_canonicalize").run(res)
                        if not (res.failures or res.errors):
                            pruned.add(_CANONICALIZE_TEST_ID)
                    if check_merge:
                        try:
                            if hasattr(opt_cls, "setUpClass"):
                                opt_cls.setUpClass()
                                opt_setup_ok = True
                            res = unittest.TestResult()
                            opt_cls("test_merge_subqueries").run(res)
                            if not (res.failures or res.errors):
                                pruned.add(_MERGE_SUBQ_TEST_ID)
                        except BaseException:
                            pass
                except BaseException:
                    pass

            if check_schema_cache:
                try:
                    gc.enable()
                    smod = importlib.import_module("tests.test_schema")
                    scls = smod.TestSchema
                    res = unittest.TestResult()
                    scls("test_cache_does_not_retain_queries").run(res)
                    if not (res.failures or res.errors):
                        pruned.add(_SCHEMA_CACHE_TEST_ID)
                except BaseException:
                    pass
                finally:
                    gc.disable()

            if check_agg_order:
                try:
                    from sqlglot.optimizer import optimize as _opt_agg
                    from sqlglot.planner import Plan as _Plan_agg
                    _p = _Plan_agg(
                        _opt_agg(
                            "SELECT AVG(bill_length_mm) AS avg_bill_length, AVG(bill_depth_mm) AS avg_bill_depth FROM penguins",
                            schema={"penguins": {"bill_length_mm": "DOUBLE", "bill_depth_mm": "DOUBLE"}},
                        )
                    )
                    if [agg.alias for agg in _p.root.aggregations] == ["avg_bill_length", "avg_bill_depth"]:
                        pruned.add(_AGG_ORDER_TEST_ID)
                except BaseException:
                    pass

            # --- Stage B: Differential TPC-DS Plan-Equivalence ---
            if check_tpcds:
                try:
                    import hashlib
                    from sqlglot import parse_one, transpile
                    from sqlglot.optimizer import optimize
                    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

                    _pairs = [
                        (i, m, s)
                        for i, (m, s, _) in enumerate(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
                        if m.get("execute")
                    ]
                    diff_queries: list[tuple[int, str]] = []
                    aborted_tpcds = False
                    for idx_q, _, sql_q in _tpcds_query_order(_pairs, base_by_idx):
                        expected = base_by_idx.get(idx_q)
                        if expected is None:
                            aborted_tpcds = True
                            break
                        opt_sql = optimize(parse_one(sql_q), TPCDS_SCHEMA, leave_tables_isolated=True).sql()
                        oh = hashlib.blake2b(opt_sql.encode("utf-8"), digest_size=16).hexdigest()
                        if oh != expected[0] or hashlib.blake2b(
                            transpile(sql_q, write="duckdb")[0].encode("utf-8"), digest_size=16
                        ).hexdigest() != expected[1]:
                            diff_queries.append((idx_q, sql_q))
                            if len(diff_queries) > max_diff_exec:
                                aborted_tpcds = True
                                break

                    if not aborted_tpcds:
                        if not diff_queries:
                            pruned.add(_TPCDS_TEST_ID)
                        else:
                            import duckdb
                            import numpy as np
                            import pandas as pd
                            from pandas.testing import assert_frame_equal
                            from sqlglot.executor import execute
                            from sqlglot.optimizer.scope import find_tables
                            from tests.test_executor import DIR_TPCDS

                            conn = duckdb.connect()
                            loaded_tables: dict = {}
                            for _, sql_q in diff_queries:
                                expr = parse_one(sql_q)
                                needed = {t.name for t in find_tables(expr)}
                                for tname in needed:
                                    if tname in loaded_tables:
                                        continue
                                    cols = TPCDS_SCHEMA[tname]
                                    csv_path = f"{DIR_TPCDS}{tname}.csv.gz"
                                    col_defs = ", ".join(f"{c} {ty}" for c, ty in cols.items())
                                    conn.execute(f"CREATE TABLE {tname} ({col_defs})")
                                    conn.execute(f"COPY {tname} FROM '{csv_path}' (DELIMITER '|', NULL '')")
                                    rows = conn.execute(f"SELECT * FROM {tname}").fetchall()
                                    loaded_tables[tname] = [dict(zip(cols, r)) for r in rows]
                                q_tables = {t.name: loaded_tables[t.name] for t in find_tables(expr)}
                                table = execute(expr, schema=TPCDS_SCHEMA, tables=q_tables)
                                a = conn.execute(transpile(sql_q, write="duckdb")[0]).fetchdf()
                                b = pd.DataFrame(
                                    ((np.nan if c is None else c for c in r) for r in table.rows),
                                    columns=table.columns,
                                )
                                assert_frame_equal(a, b, check_dtype=False, check_index_type=False)
                            pruned.add(_TPCDS_TEST_ID)
                except BaseException:
                    pass

            # --- Stage C: Full-commit zero-out in the SAME already-imported child! ---
            rem_items = [(tid, sec) for tid, sec in all_items if tid not in pruned]
            rem_sum = sum(sec for _, sec in rem_items)
            if (
                rem_items
                and len(rem_items) <= max_tests
                and rem_sum <= max_total_seconds
                and not any(
                    tid.startswith(_EXCLUDED_PREFIXES)
                    or tid in _EXCLUDED_IDS
                    or (touches_pickle_risk and tid.startswith("tests/test_optimizer.py::"))
                    for tid, _ in rem_items
                )
            ):
                ordered = sorted(rem_items, key=lambda x: x[1])
                loaded_mods: dict[str, object] = {}
                setup_classes: set[type] = set()
                failed_classes: set[type] = set()
                if opt_cls is not None and opt_setup_ok:
                    setup_classes.add(opt_cls)
                for idx_r, (tid, _sec) in enumerate(ordered):
                    parts = tid.split("::")
                    if len(parts) != 3:
                        break
                    tfile, cn, mn = parts
                    mod_name = tfile[:-3].replace("/", ".")
                    if mod_name not in loaded_mods:
                        try:
                            m = importlib.import_module(mod_name)
                            if mod_name == "tests.test_optimizer":
                                _patch_optimizer_pool(m, changed_dialects)
                            loaded_mods[mod_name] = m
                        except BaseException:
                            loaded_mods[mod_name] = None
                    m = loaded_mods[mod_name]
                    if m is None:
                        if sum(s for _, s in ordered[idx_r + 1 :]) <= early_abort_rem_sec:
                            break
                        continue
                    cls = getattr(m, cn, None)
                    if cls is None or not isinstance(cls, type) or not issubclass(cls, unittest.TestCase):
                        continue
                    if cls not in setup_classes and cls not in failed_classes:
                        if cn == "TestExecutor":
                            setup_classes.add(cls)
                        elif hasattr(cls, "setUpClass"):
                            try:
                                cls.setUpClass()
                                setup_classes.add(cls)
                            except BaseException:
                                failed_classes.add(cls)
                        else:
                            setup_classes.add(cls)
                    if cls in failed_classes:
                        if sum(s for _, s in ordered[idx_r + 1 :]) <= early_abort_rem_sec:
                            break
                        continue
                    res = unittest.TestResult()
                    try:
                        cls(mn).run(res)
                    except BaseException:
                        if sum(s for _, s in ordered[idx_r + 1 :]) <= early_abort_rem_sec:
                            break
                        continue
                    if not (res.failures or res.errors):
                        pruned.add(tid)
                    else:
                        if sum(s for _, s in ordered[idx_r + 1 :]) <= early_abort_rem_sec:
                            break

            os.write(w_fd, json.dumps({"pruned": sorted(pruned)}).encode("utf-8"))
        except BaseException:
            try:
                os.write(w_fd, json.dumps({"pruned": sorted(pruned)}).encode("utf-8"))
            except BaseException:
                pass
        finally:
            try:
                os.close(w_fd)
            except OSError:
                pass
            os._exit(0)

    os.close(w_fd)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(r_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(r_fd)
        try:
            _, status = os.waitpid(pid, 0)
        except ChildProcessError:
            status = 1

    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0 or not chunks:
        return selected
    try:
        pruned_set = set(json.loads(b"".join(chunks).decode("utf-8")).get("pruned", ()))
    except BaseException:
        return selected
    if not pruned_set:
        return selected
    return [t for t in selected if t["id"] not in pruned_set]
