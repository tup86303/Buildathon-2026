"""Fast, read-only-safe subprocess verifier for small fast commits using a sitecustomize.py
MetaPathFinder overlay (zero full-repo copy, works across multiprocessing spawn workers)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SITECUSTOMIZE_TMPL = """import importlib.abc, importlib.machinery, importlib.util, json, os, sys
_OV = json.loads(os.environ.get("SQLGLOT_PATCH_OVERLAY", "{}"))
_ROOT = os.environ.get("SQLGLOT_SRC_ROOT", "")
class _OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in _OV:
            patched_file, orig_file = _OV[fullname]
            class _Loader(importlib.machinery.SourceFileLoader):
                def get_data(self, p):
                    with open(patched_file, "rb") as f:
                        return f.read()
            return importlib.util.spec_from_file_location(fullname, orig_file, loader=_Loader(fullname, orig_file))
        return None
if _OV:
    sys.meta_path.insert(0, _OverlayFinder())
"""


def _rel_to_mod(rel: str) -> str:
    return rel[:-3].replace("/", ".")


def verify_fast_tests(
    changes: list[dict],
    selected: list[dict],
    index,
    sqlglot_root: Path,
    verify_script: Path,
    max_total_seconds: float = 0.70,
    max_tests: int = 10,
    disk_read_files: set[str] | None = None,
) -> list[dict]:
    if not selected or len(selected) > max_tests or not sqlglot_root.is_dir() or not verify_script.is_file():
        return selected
    # Never run multiprocessing-heavy test files (test_optimizer.py / test_executor.py / leakcheck.py)
    if any(t["id"].startswith(("tests/test_optimizer.py::", "tests/test_executor.py::", "tests/leakcheck.py::", "tests/test_sqlglotc.py::")) for t in selected):
        return selected
    total_sec = sum(t["seconds"] for t in selected)
    if total_sec > max_total_seconds:
        return selected
    # Only verify pure sqlglot/*.py edits (no tests/*, no __init__.py, no rename/delete/new)
    patched: dict[str, bytes] = {}
    for ch in changes:
        old_p = ch.get("old", "")
        new_p = ch.get("new", "")
        if (
            not old_p or old_p != new_p
            or not old_p.startswith("sqlglot/")
            or not old_p.endswith(".py")
            or Path(old_p).name == "__init__.py"
            or ch.get("is_deleted") or ch.get("is_new") or ch.get("is_rename")
            or (disk_read_files is not None and old_p in disk_read_files)
        ):
            return selected
        raw = index.reconstruct(ch)
        if raw is None:
            return selected
        patched[old_p] = raw
    if not patched:
        return selected

    tids = [t["id"] for t in selected]
    with tempfile.TemporaryDirectory(prefix="v7ov_", dir="/tmp") as td:
        td_path = Path(td)
        (td_path / "sitecustomize.py").write_text(SITECUSTOMIZE_TMPL, encoding="utf-8")
        ov_map: dict[str, list[str]] = {}
        for i, (rel, raw) in enumerate(patched.items()):
            pf = td_path / f"p_{i}.py"
            pf.write_bytes(raw)
            ov_map[_rel_to_mod(rel)] = [str(pf), str(sqlglot_root / rel)]
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{td_path}:{sqlglot_root}"
        env["SQLGLOT_PATCH_OVERLAY"] = json.dumps(ov_map)
        env["SQLGLOT_SRC_ROOT"] = str(sqlglot_root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, "-B", str(verify_script), str(sqlglot_root), json.dumps(tids)],
                env=env,
                capture_output=True,
                text=True,
                timeout=max(3.0, total_sec * 4.0 + 2.0),
            )
        except BaseException:
            return selected
        if proc.returncode != 0 or not proc.stdout.strip():
            return selected
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except BaseException:
            return selected
        if data.get("ran") != len(tids):
            return selected
        failed_set = set(data.get("failed", ()))
        return [t for t in selected if t["id"] in failed_set]
