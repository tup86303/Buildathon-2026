"""Collect runtime operand types from the clean, pinned test suite. No test IDs or outcomes."""
import ast
import atexit
import builtins
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from importlib.machinery import PathFinder, SourceFileLoader
import json
from multiprocessing.util import Finalize
import os
from pathlib import Path
import sys

observed = defaultdict(set)
process = None

@lru_cache(maxsize=None)
def describe(cls):
    family = next((base.__name__ for base in (object, int, str)
                   if cls.__eq__ is base.__eq__ and cls.__ne__ is base.__ne__), "custom")
    return (cls.__module__ + "." + cls.__qualname__, family, issubclass(cls, Enum))

def save():
    if observed and process == os.getpid():
        path = Path(os.environ["OPERAND_PROFILE_DIR"]) / f"{process}.json"
        path.write_text(json.dumps({key: sorted(values) for key, values in observed.items()}))

def operand(key, value):
    global process
    if process != os.getpid():
        observed.clear()
        process = os.getpid()
        Finalize(None, save, exitpriority=0)
    observed[key].add(describe(type(value)))
    return value

class Watch(ast.NodeTransformer):
    def __init__(self, path):
        self.path = path

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
            return node
        lhs, rhs = node.left, node.comparators[0]
        if not (isinstance(lhs, ast.Attribute) and isinstance(lhs.value, ast.Name) and lhs.value.id == "self"):
            return node
        if not ((isinstance(rhs, ast.Constant) and rhs.value is None) or
                (isinstance(rhs, ast.Attribute) and isinstance(rhs.value, ast.Name) and rhs.attr.isupper())):
            return node
        key = f"{self.path}:{node.lineno}:{node.col_offset}"
        def wrap(side, child):
            return ast.copy_location(ast.Call(ast.Name("__rwx_operand__", ast.Load()),
                                              [ast.Constant(key + side), child], []), child)
        node.left = wrap("L", lhs)
        node.comparators[0] = wrap("R", rhs)
        return node

class ProfileLoader(SourceFileLoader):
    def get_code(self, fullname):
        filename = self.get_filename(fullname)
        rel = Path(filename).relative_to(os.environ["SQLGLOT_ROOT"]).as_posix()
        tree = Watch(rel).visit(ast.parse(self.get_data(filename), filename))
        return compile(ast.fix_missing_locations(tree), filename, "exec", dont_inherit=True)

class Finder:
    @staticmethod
    def find_spec(fullname, path, target=None):
        if fullname == "sqlglot" or fullname.startswith("sqlglot."):
            spec = PathFinder.find_spec(fullname, path)
            if spec and isinstance(spec.loader, SourceFileLoader):
                spec.loader = ProfileLoader(fullname, spec.origin)
                return spec
        return None

def install():
    builtins.__rwx_operand__ = operand
    sys.meta_path.insert(0, Finder())
    atexit.register(save)

if __name__ == "__main__":
    import gzip
    merged = defaultdict(set)
    for file in Path(sys.argv[1]).glob("*.json"):
        for key, values in json.loads(file.read_bytes()).items():
            merged[key].update(map(tuple, values))
    if not merged:
        raise ValueError("empty operand profile")
    data = json.dumps({k: sorted(v) for k, v in merged.items()}, sort_keys=True, separators=(",", ":"))
    Path(sys.argv[2]).write_bytes(gzip.compress(data.encode(), mtime=0))
