"""Build a per-dialect runtime usage profile (Dialect, Tokenizer, Parser, Generator,
per-attribute scalar reads, TrackedDict key reads, and lazy_import_lines) from the clean,
pinned sqlglot test suite. Zero hardcoded test names."""

from __future__ import annotations

import ast
import concurrent.futures
import enum
import gzip
import importlib
import json
import os
import sys
import time
import unittest
from collections import defaultdict
from pathlib import Path

TRACKED_DICT_SPECS = (
    ("tokenizer_class", "NUMERIC_ESCAPES"),
    ("parser_class", "FUNCTIONS"),
    ("parser_class", "FUNCTION_PARSERS"),
    ("parser_class", "PROPERTY_PARSERS"),
    ("parser_class", "SHOW_PARSERS"),
    ("generator_class", "TYPE_MAPPING"),
    ("generator_class", "TRANSFORMS"),
    ("generator_class", "PROPERTIES_LOCATION"),
    ("parser_class", "STATEMENT_PARSERS"),
    ("parser_class", "RANGE_PARSERS"),
    ("parser_class", "EXPRESSION_PARSERS"),
    ("parser_class", "NO_PAREN_FUNCTIONS"),
    ("parser_class", "NO_PAREN_FUNCTION_PARSERS"),
    ("parser_class", "CONSTRAINT_PARSERS"),
    ("dialect", "COLUMN_ALIAS_PRECEDENCE"),
    ("dialect", "EXPRESSION_METADATA"),
)
TRACKED_DICT_ATTRS = sorted({attr for _, attr in TRACKED_DICT_SPECS})


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


def _path_to_module(rel_path: str) -> str:
    p = Path(rel_path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _dialect_rels_from_mro(obj, base_cls) -> set[str]:
    cls = obj if isinstance(obj, type) else type(obj)
    out = set()
    for base in cls.__mro__:
        if base is base_cls or base is object:
            continue
        if issubclass(base, base_cls):
            mod = getattr(base, "__module__", "")
            if mod.startswith(("sqlglot.dialects.", "sqlglot.parsers.", "sqlglot.generators.")):
                stem = mod.rsplit(".", 1)[-1]
                if stem not in {"dialect", "parser", "generator", "__init__"}:
                    out.add(f"sqlglot/dialects/{stem}.py")
    return out


def _is_scalar_const(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (bool, str, int, float)) or node.value is None
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return True
    return False


def _discover_pure_runtime_dialect_attrs(sqlglot_root: Path) -> set[str]:
    """Collect uppercase scalar attrs defined directly on Dialect subclasses,
    excluding any uppercase attr read inside _Dialect.__new__ at class creation time."""
    new_attrs: set[str] = set()
    dialect_py = sqlglot_root / "sqlglot" / "dialects" / "dialect.py"
    if dialect_py.is_file():
        tree = ast.parse(dialect_py.read_bytes())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "_Dialect":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__new__":
                        for n in ast.walk(item):
                            if isinstance(n, ast.Attribute) and n.attr.isupper():
                                new_attrs.add(n.attr)
    watch: set[str] = set()
    for f in sorted((sqlglot_root / "sqlglot" / "dialects").glob("*.py")):
        if f.name in {"dialect.py", "__init__.py"}:
            continue
        tree = ast.parse(f.read_bytes())
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and _is_scalar_const(stmt.value):
                        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                        for t in targets:
                            if isinstance(t, ast.Name) and t.id.isupper() and t.id not in new_attrs:
                                watch.add(t.id)
    # Also exclude any attr read at module/class scope outside def/lambda (e.g. Presto.TIME_FORMAT = MySQL.TIME_FORMAT)
    import_read: set[str] = set()
    for f in sorted((sqlglot_root / "sqlglot").rglob("*.py")):
        t = ast.parse(f.read_bytes())
        def _scan(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in watch:
                import_read.add(node.attr)
            for ch in ast.iter_child_nodes(node):
                _scan(ch)
        _scan(t)
    return watch - import_read


def _key_token(k) -> str:
    if isinstance(k, enum.Enum):
        return f"enum:{type(k).__name__}.{k.name}"
    if isinstance(k, bool):
        return f"const:{json.dumps(k)}"
    if isinstance(k, (str, int)):
        return f"const:{json.dumps(k)}"
    if isinstance(k, type):
        return f"type:{k.__name__}"
    return "*"


def build(sqlglot_root: Path, tests_json: Path, out_path: Path) -> None:
    t0 = time.perf_counter()
    os.chdir(sqlglot_root)
    os.environ["SKIP_INTEGRATION"] = "0"
    if str(sqlglot_root) not in sys.path:
        sys.path.insert(0, str(sqlglot_root))

    concurrent.futures.ProcessPoolExecutor = _SyncPool

    watch_attrs = _discover_pure_runtime_dialect_attrs(sqlglot_root)

    from sqlglot.dialects.dialect import Dialect, _Dialect
    from sqlglot.tokens import Tokenizer
    from sqlglot.parser import Parser
    from sqlglot.generator import Generator

    # Import all dialects so we can wrap verified runtime dicts before any tests run.
    for f in sorted((sqlglot_root / "sqlglot" / "dialects").glob("*.py")):
        if f.name in {"dialect.py", "__init__.py"}:
            continue
        try:
            importlib.import_module(f"sqlglot.dialects.{f.stem}")
        except BaseException:
            pass

    cur_d: set[str] = set()
    cur_tok: set[str] = set()
    cur_par: set[str] = set()
    cur_gen: set[str] = set()
    cur_attr: set[tuple[str, str]] = set()
    cur_dict: set[tuple[str, str, str]] = set()
    tracking_active = False

    class _TrackedDict(dict):
        __slots__ = ("_owners", "_attr")
        def __init__(self, d: dict, owners: tuple[str, ...], attr: str):
            super().__init__(d)
            self._owners = owners
            self._attr = attr
        def get(self, key, default=None):
            if tracking_active:
                tok = _key_token(key)
                for o in self._owners:
                    cur_dict.add((o, self._attr, tok))
            return super().get(key, default)
        def __getitem__(self, key):
            if tracking_active:
                tok = _key_token(key)
                for o in self._owners:
                    cur_dict.add((o, self._attr, tok))
            return super().__getitem__(key)
        def _mark_all(self):
            if tracking_active:
                for o in self._owners:
                    cur_dict.add((o, self._attr, "*"))
        def items(self):
            self._mark_all()
            return super().items()
        def values(self):
            self._mark_all()
            return super().values()
        def copy(self):
            self._mark_all()
            return super().copy()
        def __eq__(self, other):
            self._mark_all()
            return super().__eq__(other)
        def __ne__(self, other):
            self._mark_all()
            return super().__ne__(other)
        def __repr__(self):
            self._mark_all()
            return super().__repr__()

    def _all_subclasses(base):
        seen = set()
        stack = list(base.__subclasses__())
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                stack.extend(c.__subclasses__())
        return seen

    # Import all parser/generator modules too so auxiliary classes (e.g. AthenaTrinoParser) are loaded
    for sub in ("parsers", "generators"):
        for f in sorted((sqlglot_root / "sqlglot" / sub).glob("*.py")):
            if f.name == "__init__.py":
                continue
            try:
                importlib.import_module(f"sqlglot.{sub}.{f.stem}")
            except BaseException:
                pass

    wrapped_dict_pairs: set[tuple[str, str]] = set()
    base_by_kind = {
        "dialect": Dialect,
        "tokenizer_class": Tokenizer,
        "parser_class": Parser,
        "generator_class": Generator,
    }
    for holder_kind, attr in TRACKED_DICT_SPECS:
        base_cls = base_by_kind[holder_kind]
        holders = set(_all_subclasses(base_cls))
        if holder_kind == "dialect":
            holders |= set(Dialect.classes.values())
        else:
            for dcls in set(Dialect.classes.values()):
                h = getattr(dcls, holder_kind, None)
                if h is not None:
                    holders.add(h)
        for holder in holders:
            owners = tuple(sorted(_dialect_rels_from_mro(holder, base_cls)))
            if not owners:
                continue
            val = getattr(holder, attr, None)
            if isinstance(val, dict) and not isinstance(val, _TrackedDict):
                try:
                    setattr(holder, attr, _TrackedDict(val, owners, attr))
                except BaseException:
                    continue
            if isinstance(getattr(holder, attr, None), _TrackedDict):
                for o in owners:
                    wrapped_dict_pairs.add((o, attr))

    # Wrap Generator _DISPATCH_CACHE for ALL Generator subclasses (including auxiliary ones like AthenaTrinoGenerator)
    from sqlglot.generator import _DISPATCH_CACHE, _build_dispatch
    _DISPATCH_CACHE.clear()
    all_gens = set(_all_subclasses(Generator)) | {
        getattr(dcls, "generator_class", None) for dcls in set(Dialect.classes.values())
    }
    for gen_cls in all_gens:
        if gen_cls is None:
            continue
        owners = tuple(sorted(_dialect_rels_from_mro(gen_cls, Generator)))
        if owners:
            _DISPATCH_CACHE[gen_cls] = _TrackedDict(_build_dispatch(gen_cls), owners, "TRANSFORMS")
            for o in owners:
                wrapped_dict_pairs.add((o, "TRANSFORMS"))

    # Clear any cached dialect instances so fresh instances see wrapped dicts
    if hasattr(Dialect, "_instances"):
        Dialect._instances.clear()

    orig_gor = Dialect.get_or_raise.__func__
    orig_d_init = Dialect.__init__
    orig_tok_init = Tokenizer.__init__
    orig_par_init = Parser.__init__
    orig_gen_init = Generator.__init__
    orig_d_get = Dialect.__getattribute__
    orig_dm_get = _Dialect.__getattribute__

    @classmethod
    def _hook_gor(cls, dialect):
        res = orig_gor(cls, dialect)
        if tracking_active and res is not None:
            cur_d.update(_dialect_rels_from_mro(res, Dialect))
        return res

    def _hook_d(self, *a, **kw):
        if tracking_active:
            cur_d.update(_dialect_rels_from_mro(self, Dialect))
        return orig_d_init(self, *a, **kw)

    def _hook_tok(self, *a, **kw):
        if tracking_active:
            rels = _dialect_rels_from_mro(self, Tokenizer)
            d = getattr(self, "dialect", None) or kw.get("dialect")
            if d is not None:
                rels.update(_dialect_rels_from_mro(d, Dialect))
            cur_tok.update(rels)
        return orig_tok_init(self, *a, **kw)

    def _hook_par(self, *a, **kw):
        if tracking_active:
            rels = _dialect_rels_from_mro(self, Parser)
            d = getattr(self, "dialect", None) or kw.get("dialect")
            if d is not None:
                rels.update(_dialect_rels_from_mro(d, Dialect))
            cur_par.update(rels)
        return orig_par_init(self, *a, **kw)

    def _hook_gen(self, *a, **kw):
        if tracking_active:
            rels = _dialect_rels_from_mro(self, Generator)
            d = getattr(self, "dialect", None) or kw.get("dialect")
            if d is not None:
                rels.update(_dialect_rels_from_mro(d, Dialect))
            cur_gen.update(rels)
        return orig_gen_init(self, *a, **kw)

    def _hook_d_get(self, name):
        if tracking_active and name in watch_attrs:
            for rel in _dialect_rels_from_mro(self, Dialect):
                cur_attr.add((rel, name))
        return orig_d_get(self, name)

    def _hook_dm_get(cls, name):
        if tracking_active and name in watch_attrs:
            for rel in _dialect_rels_from_mro(cls, Dialect):
                cur_attr.add((rel, name))
        return orig_dm_get(cls, name)

    Dialect.get_or_raise = _hook_gor
    Dialect.__init__ = _hook_d
    Tokenizer.__init__ = _hook_tok
    Parser.__init__ = _hook_par
    Generator.__init__ = _hook_gen
    Dialect.__getattribute__ = _hook_d_get
    _Dialect.__getattribute__ = _hook_dm_get

    tests = json.loads(tests_json.read_text(encoding="utf-8"))["tests"]
    by_file: dict[str, list[tuple[str, str, str, float]]] = defaultdict(list)
    for t in tests:
        parts = t["id"].split("::")
        if len(parts) == 3:
            by_file[parts[0]].append((parts[1], parts[2], t["id"], t["seconds"]))

    d_map: dict[str, set[str]] = defaultdict(set)
    tok_map: dict[str, set[str]] = defaultdict(set)
    par_map: dict[str, set[str]] = defaultdict(set)
    gen_map: dict[str, set[str]] = defaultdict(set)
    attr_map: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    dict_map: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    unprofiled: list[str] = []
    attr_unprofiled: list[str] = []

    devnull = open(os.devnull, "w")
    old_o, old_e = sys.stdout, sys.stderr
    try:
        for tfile, items in sorted(by_file.items()):
            mod_name = _path_to_module(tfile)
            sys.stdout = sys.stderr = devnull
            try:
                mod = importlib.import_module(mod_name)
            except BaseException:
                unprofiled.extend(tid for _, _, tid, _ in items)
                continue
            by_cls: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
            for cn, mn, tid, sec in items:
                by_cls[cn].append((mn, tid, sec))
            for cn, meths in by_cls.items():
                cls = getattr(mod, cn, None)
                if cls is None or not isinstance(cls, type) or not issubclass(cls, unittest.TestCase):
                    unprofiled.extend(tid for _, tid, _ in meths)
                    continue
                s_d, s_tok, s_par, s_gen = set(), set(), set(), set()
                s_attr: set[tuple[str, str]] = set()
                s_dict: set[tuple[str, str, str]] = set()
                if hasattr(cls, "setUpClass"):
                    cur_d.clear(); cur_tok.clear(); cur_par.clear(); cur_gen.clear()
                    cur_attr.clear(); cur_dict.clear()
                    tracking_active = True
                    try:
                        cls.setUpClass()
                    except BaseException:
                        tracking_active = False
                        unprofiled.extend(tid for _, tid, _ in meths)
                        continue
                    tracking_active = False
                    s_d, s_tok, s_par, s_gen = set(cur_d), set(cur_tok), set(cur_par), set(cur_gen)
                    s_attr, s_dict = set(cur_attr), set(cur_dict)
                for mn, tid, sec in meths:
                    if os.environ.get("SKIP_SLOW_PROFILE") == "1" and sec > 15.0:
                        for r in s_d: d_map[r].add(tid)
                        for r in s_tok: tok_map[r].add(tid)
                        for r in s_par: par_map[r].add(tid)
                        for r in s_gen: gen_map[r].add(tid)
                        attr_unprofiled.append(tid)
                        continue
                    cur_d.clear(); cur_tok.clear(); cur_par.clear(); cur_gen.clear()
                    cur_attr.clear(); cur_dict.clear()
                    tracking_active = True
                    try:
                        result = unittest.TestResult()
                        cls(mn).run(result)
                        if result.testsRun != 1 or not result.wasSuccessful() or result.skipped:
                            unprofiled.append(tid)
                            continue
                    except BaseException:
                        unprofiled.append(tid)
                        continue
                    finally:
                        tracking_active = False
                    for r in cur_d | s_d: d_map[r].add(tid)
                    for r in cur_tok | s_tok: tok_map[r].add(tid)
                    for r in cur_par | s_par: par_map[r].add(tid)
                    for r in cur_gen | s_gen: gen_map[r].add(tid)
                    for r, a in cur_attr | s_attr: attr_map[r][a].add(tid)
                    for r, a, k in cur_dict | s_dict: dict_map[r][a][k].add(tid)
                if hasattr(cls, "tearDownClass"):
                    try: cls.tearDownClass()
                    except BaseException: pass
    finally:
        tracking_active = False
        Dialect.__getattribute__ = orig_d_get
        _Dialect.__getattribute__ = orig_dm_get
        sys.stdout = old_o
        sys.stderr = old_e
        devnull.close()

    # Trace core lines executed during lazy import of dialects with import-time exp.maybe_parse()
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
        try: importlib.import_module(f"sqlglot.dialects.{dname}")
        except BaseException: pass
        finally: sys.settrace(None)
        lazy_import_lines[drel] = {k: sorted(v) for k, v in sorted(hit_lines.items())}

    # Ensure every wrapped (d_rel, attr) pair has a key in dict_key_tests even if empty
    dict_payload: dict[str, dict[str, dict[str, list[str]]]] = {}
    for d_rel, attr in sorted(wrapped_dict_pairs):
        dict_payload.setdefault(d_rel, {})[attr] = {
            k: sorted(v) for k, v in sorted(dict_map.get(d_rel, {}).get(attr, {}).items())
        }

    payload = {
        "dialect_tests": {k: sorted(v) for k, v in sorted(d_map.items())},
        "tokenizer_tests": {k: sorted(v) for k, v in sorted(tok_map.items())},
        "parser_tests": {k: sorted(v) for k, v in sorted(par_map.items())},
        "generator_tests": {k: sorted(v) for k, v in sorted(gen_map.items())},
        "dialect_attr_tests": {
            r: {a: sorted(tids) for a, tids in sorted(am.items())}
            for r, am in sorted(attr_map.items())
        },
        "watched_dialect_attrs": sorted(watch_attrs),
        "dict_key_tests": dict_payload,
        "tracked_dict_attrs": TRACKED_DICT_ATTRS,
        "unprofiled": sorted(set(unprofiled)),
        "attr_unprofiled": sorted(set(attr_unprofiled)),
        "lazy_import_lines": lazy_import_lines,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out_path.write_bytes(gzip.compress(raw, mtime=0))
    print(
        f"Built {out_path} in {time.perf_counter()-t0:.1f}s | "
        f"dialects={len(d_map)} | watch_attrs={len(watch_attrs)} | "
        f"attr_dialects={len(attr_map)} | dict_pairs={len(wrapped_dict_pairs)} | "
        f"unprof={len(unprofiled)} | attr_unprof={len(attr_unprofiled)}"
    )


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
