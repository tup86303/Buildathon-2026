"""Build a per-dialect runtime usage map from the clean, pinned sqlglot test suite.

For each test in tests.json, records every Dialect subclass (including all Dialect
ancestors in MRO, capturing STARTS-style class-hierarchy inheritance like
Spark -> Spark2 -> Hive) instantiated via Dialect.get_or_raise or Dialect.__init__.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import importlib
import json
import os
import sys
import time
import unittest
from collections import defaultdict
from pathlib import Path


class _SyncPool:
    """Synchronous drop-in for ProcessPoolExecutor returning real Future objects."""

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


def _path_to_module(rel_path: str) -> str:
    p = Path(rel_path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def build(sqlglot_root: Path, tests_json: Path, out_path: Path) -> None:
    t0 = time.perf_counter()
    os.chdir(sqlglot_root)
    os.environ["SKIP_INTEGRATION"] = "0"
    if str(sqlglot_root) not in sys.path:
        sys.path.insert(0, str(sqlglot_root))

    concurrent.futures.ProcessPoolExecutor = _SyncPool

    from sqlglot.dialects.dialect import Dialect

    current_dialects: set[str] = set()
    orig_get_or_raise = Dialect.get_or_raise.__func__
    orig_init = Dialect.__init__

    def _record(obj):
        cls = obj if isinstance(obj, type) else type(obj)
        for base in cls.__mro__:
            if base is Dialect or base is object:
                continue
            if issubclass(base, Dialect):
                mod = getattr(base, "__module__", "")
                if mod.startswith("sqlglot.dialects."):
                    rel = mod.replace(".", "/") + ".py"
                    current_dialects.add(rel)

    @classmethod
    def _hook_get_or_raise(cls, dialect):
        res = orig_get_or_raise(cls, dialect)
        if res is not None:
            _record(res)
        return res

    def _hook_init(self, *args, **kwargs):
        _record(self)
        return orig_init(self, *args, **kwargs)

    Dialect.get_or_raise = _hook_get_or_raise
    Dialect.__init__ = _hook_init

    tests = json.loads(tests_json.read_text(encoding="utf-8"))["tests"]
    by_file: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for t in tests:
        parts = t["id"].split("::")
        if len(parts) == 3:
            by_file[parts[0]].append((parts[1], parts[2], t["id"]))

    dialect_to_tests: dict[str, set[str]] = defaultdict(set)
    unprofiled: list[str] = []

    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        for tfile, items in sorted(by_file.items()):
            mod_name = _path_to_module(tfile)
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                mod = importlib.import_module(mod_name)
            except BaseException:
                for _, _, tid in items:
                    unprofiled.append(tid)
                continue

            by_cls: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for cls_name, meth_name, tid in items:
                by_cls[cls_name].append((meth_name, tid))

            for cls_name, meths in by_cls.items():
                cls = getattr(mod, cls_name, None)
                if cls is None or not isinstance(cls, type) or not issubclass(cls, unittest.TestCase):
                    for _, tid in meths:
                        unprofiled.append(tid)
                    continue
                setup_dialects: set[str] = set()
                if hasattr(cls, "setUpClass"):
                    current_dialects.clear()
                    try:
                        cls.setUpClass()
                    except BaseException:
                        for _, tid in meths:
                            unprofiled.append(tid)
                        continue
                    setup_dialects = set(current_dialects)
                for meth_name, tid in meths:
                    current_dialects.clear()
                    try:
                        inst = cls(meth_name)
                        res = unittest.TestResult()
                        inst.run(res)
                    except BaseException:
                        unprofiled.append(tid)
                        continue
                    for drel in current_dialects | setup_dialects:
                        dialect_to_tests[drel].add(tid)
                if hasattr(cls, "tearDownClass"):
                    try:
                        cls.tearDownClass()
                    except BaseException:
                        pass
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        devnull.close()

    # Trace core lines executed during lazy import of dialects that run exp.maybe_parse() at import time
    lazy_import_lines: dict[str, dict[str, list[int]]] = {}
    root_str = str(sqlglot_root)
    for dname in ("duckdb", "bigquery"):
        drel = f"sqlglot/dialects/{dname}.py"
        for k in list(sys.modules.keys()):
            if k in (f"sqlglot.dialects.{dname}", f"sqlglot.parsers.{dname}", f"sqlglot.generators.{dname}"):
                sys.modules.pop(k, None)
        hit_lines: dict[str, set[int]] = defaultdict(set)
        def _trace(frame, event, arg):
            if event == "line":
                fn = frame.f_code.co_filename
                if fn.startswith(root_str + "/sqlglot/"):
                    rel = fn[len(root_str) + 1:]
                    if not rel.startswith((f"sqlglot/dialects/{dname}", f"sqlglot/parsers/{dname}", f"sqlglot/generators/{dname}")):
                        hit_lines[rel].add(frame.f_lineno)
            return _trace
        sys.settrace(_trace)
        try:
            importlib.import_module(f"sqlglot.dialects.{dname}")
        except BaseException:
            pass
        finally:
            sys.settrace(None)
        lazy_import_lines[drel] = {k: sorted(v) for k, v in sorted(hit_lines.items())}

    payload = {
        "dialect_tests": {k: sorted(v) for k, v in sorted(dialect_to_tests.items())},
        "unprofiled": sorted(set(unprofiled)),
        "lazy_import_lines": lazy_import_lines,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out_path.write_bytes(gzip.compress(raw, mtime=0))
    dt = time.perf_counter() - t0
    print(f"Built {out_path} in {dt:.1f}s | dialects={len(dialect_to_tests)} | unprofiled={len(unprofiled)}")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
