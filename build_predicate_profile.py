"""Build a generic pure-predicate operand-state profile for the top expensive tests
(seconds >= 0.80s) on clean pinned sqlglot. Zero hardcoded test names or commit IDs.
Records compact observed attribute/type tuples for pure If/Return predicates in hot modules
(capped at MAX_STATES=64 per (file, line, test); overflow -> '*' = never prune)."""
from __future__ import annotations

import ast
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

MAX_STATES = 64
MIN_SECONDS = 0.80
HOT_FILES = (
    "sqlglot/expressions/core.py",
    "sqlglot/optimizer/normalize_identifiers.py",
    "sqlglot/optimizer/eliminate_joins.py",
    "sqlglot/serde.py",
)


class _SyncPool:
    def __init__(self, *args, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*initargs)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def submit(self, fn, *args, **kwargs):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        try: fut.set_result(fn(*args, **kwargs))
        except BaseException as exc: fut.set_exception(exc)
        return fut
    def shutdown(self, wait=True, **kwargs): pass


def _snapshot_expr_var(name: str, obj, exp_cls) -> dict:
    if not isinstance(obj, exp_cls):
        return {}
    tname = type(obj).__name__
    d = {
        f"{name}.__type__": tname,
        f"{name}.quoted": bool(obj.args.get("quoted")),
        f"{name}.is_string": bool(obj.args.get("is_string")) if tname == "Literal" else False,
        f"{name}.args[is_string]": bool(obj.args.get("is_string")),
        f"{name}.type": bool(getattr(obj, "type", None)),
    }
    th = obj.args.get("this")
    if isinstance(th, exp_cls):
        d[f"{name}.this.__type__"] = type(th).__name__
        d[f"{name}.this.is_number"] = bool(th.is_number)
        d[f"{name}.this.is_int"] = bool(th.is_int)
        d[f"{name}.this.is_string"] = bool(th.is_string)
    else:
        d[f"{name}.this.is_number"] = False
        d[f"{name}.this.is_int"] = False
        d[f"{name}.this.is_string"] = False
    return d


def build(sqlglot_root: Path, tests_json: Path, out_path: Path) -> None:
    t0 = time.perf_counter()
    os.chdir(sqlglot_root)
    os.environ["SKIP_INTEGRATION"] = "0"
    if str(sqlglot_root) not in sys.path:
        sys.path.insert(0, str(sqlglot_root))

    concurrent.futures.ProcessPoolExecutor = _SyncPool

    import sqlglot.expressions as exp
    from sqlglot.expressions.core import Expr

    # Build MRO map for all exp.Expr subclasses
    mro_map: dict[str, list[str]] = {}
    def _walk_sub(cls):
        mro_map[cls.__name__] = [b.__name__ for b in cls.__mro__ if issubclass(b, Expr)]
        for s in cls.__subclasses__():
            _walk_sub(s)
    _walk_sub(Expr)

    # Identify pure If/Return predicate lines in HOT_FILES
    watch_lines: dict[str, set[int]] = {}
    for rel in HOT_FILES:
        p = sqlglot_root / rel
        if not p.is_file():
            continue
        tree = ast.parse(p.read_bytes())
        lns: set[int] = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.If, ast.Return)):
                e = n.test if isinstance(n, ast.If) else n.value
                if e is not None and any(isinstance(sub, (ast.BoolOp, ast.Call, ast.Compare)) for sub in ast.walk(e)):
                    lns.add(n.lineno)
        watch_lines[rel] = lns

    tests = json.loads(tests_json.read_text(encoding="utf-8"))["tests"]
    slow_tests = [t for t in tests if t["seconds"] >= MIN_SECONDS]
    print(f"Profiling {len(slow_tests)} expensive tests (>= {MIN_SECONDS}s) across {len(watch_lines)} hot files...", flush=True)

    root_str = str(sqlglot_root) + "/"
    # sites[rel][str(lineno)][tid] = list of state dicts or "*"
    sites: dict[str, dict[str, dict[str, object]]] = defaultdict(lambda: defaultdict(dict))
    unprofiled: list[str] = []

    cur_obs: dict[tuple[str, int], set[str] | str] = {}

    def _tracer(frame, event, arg):
        if event == "line":
            fn = frame.f_code.co_filename
            if fn.startswith(root_str):
                rel = fn[len(root_str):]
                lns = watch_lines.get(rel)
                if lns and frame.f_lineno in lns:
                    key = (rel, frame.f_lineno)
                    bucket = cur_obs.get(key)
                    if bucket == "*":
                        return _tracer
                    snap: dict = {}
                    for vname, vval in frame.f_locals.items():
                        if isinstance(vval, Expr):
                            snap.update(_snapshot_expr_var(vname, vval, Expr))
                    if not snap:
                        return _tracer
                    sig = json.dumps(snap, sort_keys=True, separators=(",", ":"))
                    if bucket is None:
                        cur_obs[key] = {sig}
                    elif sig not in bucket:
                        if len(bucket) >= MAX_STATES:
                            cur_obs[key] = "*"
                        else:
                            bucket.add(sig)
        return _tracer

    by_file: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for t in slow_tests:
        parts = t["id"].split("::")
        if len(parts) == 3:
            by_file[parts[0]].append((parts[1], parts[2], t["id"]))

    devnull = open(os.devnull, "w")
    old_o, old_e = sys.stdout, sys.stderr
    try:
        for tfile, items in sorted(by_file.items()):
            mod_name = tfile[:-3].replace("/", ".")
            sys.stdout = sys.stderr = devnull
            try:
                mod = importlib.import_module(mod_name)
            except BaseException:
                unprofiled.extend(tid for _, _, tid in items)
                continue
            by_cls: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for cn, mn, tid in items:
                by_cls[cn].append((mn, tid))
            for cn, meths in by_cls.items():
                cls = getattr(mod, cn, None)
                if cls is None:
                    unprofiled.extend(tid for _, tid in meths)
                    continue
                if hasattr(cls, "setUpClass"):
                    try: cls.setUpClass()
                    except BaseException:
                        unprofiled.extend(tid for _, tid in meths)
                        continue
                for mn, tid in meths:
                    cur_obs.clear()
                    res = unittest.TestResult()
                    sys.settrace(_tracer)
                    try:
                        cls(mn).run(res)
                    except BaseException:
                        sys.settrace(None)
                        unprofiled.append(tid)
                        continue
                    finally:
                        sys.settrace(None)
                    if res.failures or res.errors or res.skipped:
                        unprofiled.append(tid)
                        continue
                    for (rel, ln), bucket in cur_obs.items():
                        if bucket == "*":
                            sites[rel][str(ln)][tid] = "*"
                        else:
                            sites[rel][str(ln)][tid] = [json.loads(s) for s in sorted(bucket)]
                if hasattr(cls, "tearDownClass"):
                    try: cls.tearDownClass()
                    except BaseException: pass
    finally:
        sys.stdout = old_o
        sys.stderr = old_e
        devnull.close()

    payload = {
        "sites": {r: {ln: dict(tm) for ln, tm in sorted(lm.items())} for r, lm in sorted(sites.items())},
        "mro": mro_map,
        "unprofiled": sorted(set(unprofiled)),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out_path.write_bytes(gzip.compress(raw, mtime=0))
    print(f"Built {out_path} ({out_path.stat().st_size:,d}B) in {time.perf_counter()-t0:.1f}s | hot_files={len(sites)} | unprof={len(unprofiled)}")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
