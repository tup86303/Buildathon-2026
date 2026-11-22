"""Run a small set of unittest.TestCase methods against an overlaid patched sqlglot
in a clean child process. Prints JSON {"ran": N, "failed": [...]} on stdout."""
from __future__ import annotations

import importlib
import json
import os
import sys
import unittest
from collections import defaultdict


def main() -> None:
    sqlglot_root = sys.argv[1]
    tids = json.loads(sys.argv[2])
    os.chdir(sqlglot_root)
    os.environ["SKIP_INTEGRATION"] = "0"
    if sqlglot_root not in sys.path:
        sys.path.insert(0, sqlglot_root)

    by_file: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for tid in tids:
        parts = tid.split("::")
        if len(parts) != 3:
            print(json.dumps({"ran": -1, "failed": tids}))
            return
        by_file[parts[0]].append((parts[1], parts[2], tid))

    failed: list[str] = []
    ran = 0
    devnull = open(os.devnull, "w")
    old_o, old_e = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = devnull
        for tfile, items in by_file.items():
            mod_name = tfile[:-3].replace("/", ".")
            try:
                mod = importlib.import_module(mod_name)
            except BaseException:
                failed.extend(tid for _, _, tid in items)
                ran += len(items)
                continue
            by_cls: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for cn, mn, tid in items:
                by_cls[cn].append((mn, tid))
            for cn, meths in by_cls.items():
                cls = getattr(mod, cn, None)
                if cls is None or not isinstance(cls, type) or not issubclass(cls, unittest.TestCase):
                    failed.extend(tid for _, tid in meths)
                    ran += len(meths)
                    continue
                if hasattr(cls, "setUpClass"):
                    try:
                        cls.setUpClass()
                    except BaseException:
                        failed.extend(tid for _, tid in meths)
                        ran += len(meths)
                        continue
                for mn, tid in meths:
                    res = unittest.TestResult()
                    try:
                        cls(mn).run(res)
                    except BaseException:
                        failed.append(tid)
                        ran += 1
                        continue
                    ran += 1
                    if res.failures or res.errors:
                        failed.append(tid)
                if hasattr(cls, "tearDownClass"):
                    try:
                        cls.tearDownClass()
                    except BaseException:
                        pass
    finally:
        sys.stdout = old_o
        sys.stderr = old_e
        devnull.close()

    print(json.dumps({"ran": ran, "failed": failed}))


if __name__ == "__main__":
    main()
