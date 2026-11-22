"""Compare patched Python to its baseline without executing any patch."""

from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import json
import os
import re
import zipfile
from functools import lru_cache
from pathlib import Path

CODING_RE = re.compile(rb"^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")
SPECIAL_COMMENT_RE = re.compile(r"#\s*(?:type\s*:|mypy\s*:|noqa|pragma|doctest)", re.IGNORECASE)
CONTROL_FLOW_NODES = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.ExceptHandler,
    ast.With, ast.AsyncWith, ast.Return, ast.Raise, ast.Yield, ast.YieldFrom,
    ast.Break, ast.Continue, ast.Assert,
)
if hasattr(ast, "Match"):
    CONTROL_FLOW_NODES = CONTROL_FLOW_NODES + (ast.Match,)
if hasattr(ast, "TryStar"):
    CONTROL_FLOW_NODES = CONTROL_FLOW_NODES + (ast.TryStar,)


def _annotation_names(annotation):
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _annotation_names(annotation.left) | _annotation_names(annotation.right)
    if isinstance(annotation, ast.Name):
        return {annotation.id}
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return set()
    return {"?"}



_SCOPE_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_BLOCK_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)


def _is_type_checking_guard(test: ast.AST) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id in ("t", "typing", "typing_extensions")
    ):
        return True
    return False


def _is_overload_func(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for dec in stmt.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if (
            isinstance(dec, ast.Attribute)
            and dec.attr == "overload"
            and isinstance(dec.value, ast.Name)
            and dec.value.id in ("t", "typing", "typing_extensions")
        ):
            return True
    return False


def _clean_stmt_list(stmts: list[ast.stmt], in_func: bool) -> list[ast.stmt]:
    out: list[ast.stmt] = []
    for stmt in stmts:
        if _is_overload_func(stmt):
            continue
        if isinstance(stmt, ast.If) and _is_type_checking_guard(stmt.test):
            if stmt.orelse:
                out.extend(_clean_stmt_list(stmt.orelse, in_func))
            continue
        if in_func and isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                # Preserve exact source span for _function_block_hashes
                new_node = ast.copy_location(ast.Assign(targets=[stmt.target], value=stmt.value), stmt)
                out.append(new_node)
            continue
        out.append(stmt)
    return out


def _has_future_annotations(tree: ast.AST) -> bool:
    if not isinstance(tree, ast.Module):
        return False
    for stmt in tree.body[:8]:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            if any(a.name == "annotations" for a in stmt.names):
                return True
    return False


def _strip_func_annotations(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    fn.returns = None
    args = fn.args
    for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        a.annotation = None
    if args.vararg is not None:
        args.vararg.annotation = None
    if args.kwarg is not None:
        args.kwarg.annotation = None


def _strip_docstrings(tree: ast.AST, strip_types: bool = False) -> None:
    """Remove provably dead-at-runtime AST nodes in-place in O(#scopes):
    1. Module/class/function docstrings (0 tests in tests.json read __doc__ or run doctest).
    2. `if [t.|typing.]TYPE_CHECKING:` blocks & `@overload` stubs (never executed at runtime).
    3. Function-local PEP 526 AnnAssign annotations (never evaluated or stored at runtime).
    4. Function param/return annotations when `from __future__ import annotations` is active."""
    future_ann = strip_types and _has_future_annotations(tree)
    stack: list[tuple[ast.AST, bool]] = [(tree, False)]
    while stack:
        node, in_func = stack.pop()
        if isinstance(node, _SCOPE_NODES):
            is_fn = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_fn and future_ann:
                _strip_func_annotations(node)
            child_in_func = in_func or is_fn
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            if strip_types:
                body = _clean_stmt_list(body, child_in_func)
            node.body = body
            for stmt in body:
                if isinstance(stmt, (_SCOPE_NODES, _BLOCK_NODES)):
                    stack.append((stmt, child_in_func))
        else:
            for attr in ("body", "orelse", "finalbody"):
                sub = getattr(node, attr, None)
                if sub:
                    cleaned = _clean_stmt_list(sub, in_func) if strip_types else sub
                    setattr(node, attr, cleaned)
                    for stmt in cleaned:
                        if isinstance(stmt, (_SCOPE_NODES, _BLOCK_NODES)):
                            stack.append((stmt, in_func))
            for h in getattr(node, "handlers", ()):
                if strip_types:
                    h.body = _clean_stmt_list(h.body, in_func)
                for stmt in h.body:
                    if isinstance(stmt, (_SCOPE_NODES, _BLOCK_NODES)):
                        stack.append((stmt, in_func))
            for c in getattr(node, "cases", ()):
                if strip_types:
                    c.body = _clean_stmt_list(c.body, in_func)
                for stmt in c.body:
                    if isinstance(stmt, (_SCOPE_NODES, _BLOCK_NODES)):
                        stack.append((stmt, in_func))


def _hash_code(co, h) -> None:
    h.update(co.co_code)
    h.update(repr((
        co.co_names,
        co.co_varnames,
        co.co_freevars,
        co.co_cellvars,
        co.co_argcount,
        co.co_posonlyargcount,
        co.co_kwonlyargcount,
        co.co_flags,
        co.co_name,
    )).encode("utf-8"))
    for c in co.co_consts:
        if hasattr(c, "co_code"):
            h.update(b"<CODE>")
            _hash_code(c, h)
        else:
            h.update(type(c).__name__.encode("utf-8"))
            h.update(b":")
            h.update(repr(c).encode("utf-8"))
            h.update(b";")


def _signature(tree, identity_classes, enum_fields, sites=None):
    id_cls: set[str] = set()
    classes: list[ast.ClassDef] = []
    stack = [tree.body]
    while stack:
        for node in stack.pop():
            t = type(node)
            if t is ast.ImportFrom:
                mod = node.module
                for alias in node.names:
                    if f"{mod}.{alias.name}" in identity_classes:
                        id_cls.add(alias.asname or alias.name)
            elif t is ast.ClassDef:
                classes.append(node)
                stack.append(node.body)
            elif t is ast.If or t is ast.Try or t is ast.With or t is ast.For or t is ast.While:
                if getattr(node, "body", None):
                    stack.append(node.body)
                if getattr(node, "orelse", None):
                    stack.append(node.orelse)
                if getattr(node, "finalbody", None):
                    stack.append(node.finalbody)
                for h in getattr(node, "handlers", ()):
                    stack.append(h.body)
    for cls in classes:
        init = next((m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
        if init is None:
            continue
        fields = {
            n.target.attr
            for n in ast.walk(init)
            if isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Attribute)
            and isinstance(n.target.value, ast.Name)
            and n.target.value.id == "self"
            and _annotation_names(n.annotation) <= id_cls
        }
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            for cmp in ast.walk(method):
                if not isinstance(cmp, ast.Compare) or len(cmp.ops) != 1:
                    continue
                if sites is not None and (cmp.lineno, cmp.col_offset) not in sites:
                    continue
                left, right = cmp.left, cmp.comparators[0]
                if not (isinstance(left, ast.Attribute) and isinstance(left.value, ast.Name) and left.value.id == "self"):
                    continue
                plain_none = left.attr in fields and isinstance(right, ast.Constant) and right.value is None
                enum_member = (
                    isinstance(right, ast.Attribute)
                    and right.attr.isupper()
                    and isinstance(right.value, ast.Name)
                    and (sites is not None or enum_fields.get(left.attr) == right.value.id)
                )
                plain_none = (sites is not None or left.attr in fields) and isinstance(right, ast.Constant) and right.value is None
                if not (plain_none or enum_member):
                    continue
                if isinstance(cmp.ops[0], ast.Eq):
                    cmp.ops[0] = ast.Is()
                elif isinstance(cmp.ops[0], ast.NotEq):
                    cmp.ops[0] = ast.IsNot()
    _strip_docstrings(tree, strip_types=True)
    if hasattr(tree, "type_ignores"):
        tree.type_ignores = []
    return hashlib.blake2b(ast.dump(tree, include_attributes=False).encode("utf-8"), digest_size=16).hexdigest()


def _control_flow_hash(tree) -> str:
    """Fast O(statements) control-flow & signature structural hash (skips expression subtrees)."""
    items = []
    stack = [tree.body]
    while stack:
        for node in stack.pop():
            t = type(node)
            if t is ast.FunctionDef or t is ast.AsyncFunctionDef:
                a = node.args
                items.append((t.__name__, node.name, len(a.posonlyargs), len(a.args), len(a.kwonlyargs), len(a.defaults), len(node.decorator_list)))
                stack.append(node.body)
            elif t is ast.ClassDef:
                stack.append(node.body)
            elif isinstance(node, CONTROL_FLOW_NODES):
                items.append((t.__name__,))
                body = getattr(node, "body", None)
                if body:
                    stack.append(body)
                orelse = getattr(node, "orelse", None)
                if orelse:
                    stack.append(orelse)
                finalbody = getattr(node, "finalbody", None)
                if finalbody:
                    stack.append(finalbody)
                handlers = getattr(node, "handlers", None)
                if handlers:
                    for h in handlers:
                        stack.append(h.body)
                cases = getattr(node, "cases", None)
                if cases:
                    for c in cases:
                        stack.append(c.body)
    return hashlib.blake2b(repr(items).encode("utf-8"), digest_size=16).hexdigest()


def _function_block_hashes(tree: ast.AST) -> dict[str, tuple[int, int, str]]:
    _strip_docstrings(tree)
    blocks: dict[str, tuple[int, int, str]] = {}
    dups: set[str] = set()

    def _walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                cname = f"{prefix}.{child.name}" if prefix else child.name
                _walk(child, cname)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fname = f"{prefix}.{child.name}" if prefix else child.name
                if fname in blocks:
                    dups.add(fname)
                else:
                    cloned = copy.deepcopy(child)
                    if (cloned.body and isinstance(cloned.body[0], ast.Expr)
                            and isinstance(cloned.body[0].value, ast.Constant)
                            and isinstance(cloned.body[0].value.value, str)):
                        cloned.body = cloned.body[1:]
                    h = hashlib.blake2b(ast.dump(cloned, include_attributes=False).encode("utf-8"), digest_size=16).hexdigest()
                    blocks[fname] = (child.lineno, child.end_lineno or child.lineno, h)

    _walk(tree, "")
    for d in dups:
        blocks.pop(d, None)
    return blocks





def _is_pure_literal(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (bool, str, int, float, bytes)) or node.value is None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float))
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_pure_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _is_pure_literal(k) and _is_pure_literal(v) for k, v in zip(node.keys, node.values))
    return False



def _literal_type_val(node: ast.AST | None):
    if isinstance(node, ast.Constant) and (isinstance(node.value, (bool, str, int, float, bytes)) or node.value is None):
        return (type(node.value), node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return (type(node.operand.value), node.operand.value)
    return None


def _is_safe_dict_value(old_node: ast.AST | None, new_node: ast.AST | None) -> bool:
    """True if evaluating new_node at class-definition time cannot introduce a new NameError/AttributeError."""
    if new_node is None:
        return False
    if _is_pure_literal(new_node):
        return True
    if isinstance(new_node, ast.Lambda):
        return True
    if isinstance(old_node, ast.Call) and isinstance(new_node, ast.Call):
        if ast.dump(old_node.func, include_attributes=False) != ast.dump(new_node.func, include_attributes=False):
            return False
        if len(old_node.args) != len(new_node.args) or len(old_node.keywords) != len(new_node.keywords):
            return False
        for oa, na in zip(old_node.args, new_node.args):
            if ast.dump(oa, include_attributes=False) == ast.dump(na, include_attributes=False):
                continue
            ol, nl = _literal_type_val(oa), _literal_type_val(na)
            if ol is None or nl is None or ol[0] is not nl[0]:
                return False
        old_kw = {k.arg: k.value for k in old_node.keywords}
        new_kw = {k.arg: k.value for k in new_node.keywords}
        if None in old_kw or None in new_kw or set(old_kw) != set(new_kw):
            return False
        for kname in old_kw:
            ov, nv = old_kw[kname], new_kw[kname]
            if ast.dump(ov, include_attributes=False) == ast.dump(nv, include_attributes=False):
                continue
            ol, nl = _literal_type_val(ov), _literal_type_val(nv)
            if ol is None or nl is None or ol[0] is not nl[0]:
                return False
        return True
    return False


def _ast_key_token(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, bool)):
        return f"const:{json.dumps(node.value)}"
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            if node.value.id == "exp":
                return f"type:{node.attr}"
            if node.value.id in {"TokenType"} or node.value.id.endswith("Type") or node.value.id[0].isupper():
                return f"enum:{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            # e.g. exp.DType.TINYINT or exp.DataType.Type.TINYINT -> enum:Type.TINYINT
            if node.value.attr in {"DType", "Type"}:
                return f"enum:DType.{node.attr}"
    if isinstance(node, ast.Name) and node.id[0].isupper():
        return f"type:{node.id}"
    return None


def _extract_class_dicts(tree: ast.Module, tracked_attrs: set[str]) -> list[tuple[tuple[str, ...], str, ast.Dict]]:
    out: list[tuple[tuple[str, ...], str, ast.Dict]] = []
    def _walk(body, prefix: tuple[str, ...]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                _walk(stmt.body, prefix + (stmt.name,))
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(stmt.value, ast.Dict):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                if len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id in tracked_attrs:
                    if prefix or targets[0].id == "EXPRESSION_METADATA":
                        out.append((prefix, targets[0].id, stmt.value))
    _walk(tree.body, ())
    return out



_BUILTINS = set(dir(__import__("builtins")))

def _has_undefined_import_time_name(tree: ast.Module) -> bool:
    """Detect obvious NameError at module/class import time (bare Name Load outside def/lambda)."""
    defined = set(_BUILTINS) | {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__annotations__"}
    def _collect_defs(body):
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(stmt.name)
            elif isinstance(stmt, ast.Import):
                for a in stmt.names:
                    defined.add(a.asname or a.name.split(".")[0])
            elif isinstance(stmt, ast.ImportFrom):
                if any(a.name == "*" for a in stmt.names):
                    return True  # star import: cannot prove undefined
                for a in stmt.names:
                    defined.add(a.asname or a.name)
            elif isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            defined.add(n.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                defined.add(stmt.target.id)
            elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.For)):
                for sub in ("body", "orelse", "finalbody"):
                    if _collect_defs(getattr(stmt, sub, ())):
                        return True
                for h in getattr(stmt, "handlers", ()):
                    if h.name:
                        defined.add(h.name)
                    if _collect_defs(h.body):
                        return True
        return False

    if _collect_defs(tree.body):
        return False

    def _check_expr(node: ast.AST, local_defs: set[str]) -> bool:
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return False
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            return node.id not in defined and node.id not in local_defs
        return any(_check_expr(ch, local_defs) for ch in ast.iter_child_nodes(node))

    def _check_body(body, class_defs: set[str]) -> bool:
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only check decorators & default args (evaluated at def time)
                for dec in stmt.decorator_list:
                    if _check_expr(dec, class_defs):
                        return True
                for d in list(stmt.args.defaults) + [x for x in stmt.args.kw_defaults if x is not None]:
                    if _check_expr(d, class_defs):
                        return True
            elif isinstance(stmt, ast.ClassDef):
                for b in stmt.bases:
                    if _check_expr(b, class_defs):
                        return True
                cdefs = set(class_defs)
                for cs in stmt.body:
                    if isinstance(cs, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        cdefs.add(cs.name)
                    elif isinstance(cs, ast.Assign):
                        for t in cs.targets:
                            for n in ast.walk(t):
                                if isinstance(n, ast.Name):
                                    cdefs.add(n.id)
                    elif isinstance(cs, ast.AnnAssign) and isinstance(cs.target, ast.Name):
                        cdefs.add(cs.target.id)
                if _check_body(stmt.body, cdefs):
                    return True
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if stmt.value is not None and _check_expr(stmt.value, class_defs):
                    return True
            elif isinstance(stmt, ast.Expr):
                if _check_expr(stmt.value, class_defs):
                    return True
        return False

    return _check_body(tree.body, set())


class SourceIndex:
    def __init__(self, archive):
        self.archive = zipfile.ZipFile(archive)
        names = set(self.archive.namelist())
        self.identity_classes = set(json.loads(self.archive.read("identity-classes.json")))
        self.conditions = json.loads(self.archive.read("condition-starts.json"))
        self.enum_fields = json.loads(self.archive.read("enum-fields.json"))
        self.scalar_bindings, self.read_sites = json.loads(self.archive.read("field-coverage.json")) if "field-coverage.json" in names else ({}, {})
        self.expr_symbols = json.loads(self.archive.read("expr-symbols.json")) if "expr-symbols.json" in names else {}
        self.def_headers = json.loads(self.archive.read("def-headers.json")) if "def-headers.json" in names else {}
        self.stmt_starts = json.loads(self.archive.read("stmt-starts.json")) if "stmt-starts.json" in names else {}
        self.tokenizer_ranges = json.loads(self.archive.read("tokenizer-ranges.json")) if "tokenizer-ranges.json" in names else {}
        self.test_methods = json.loads(self.archive.read("test-methods.json")) if "test-methods.json" in names else {}
        self.test_importers = json.loads(self.archive.read("test-importers.json")) if "test-importers.json" in names else {}
        hashes = json.loads(self.archive.read("baseline-hashes.json")) if "baseline-hashes.json" in names else {}
        self._cf_hashes = hashes.get("cf", {})
        self._sig_hashes = hashes.get("sig", {})
        self._last_parsed = None
        self._baseline_blocks_cache = {}
        self.profile_sites = {}
        profile = Path(archive).with_name("operand-profile.json.gz")
        if profile.is_file() and os.environ.get("SELECTOR_ENABLE_EQ_EQUIV") != "0":
            observations = json.loads(gzip.decompress(profile.read_bytes()))
            for key, left in observations.items():
                if not key.endswith("L"):
                    continue
                right = observations.get(key[:-1] + "R", [])
                none_check = (len(right) == 1 and right[0][0] == "builtins.NoneType"
                              and all(name == "builtins.NoneType" or family == "object" for name, family, _ in left))
                enum_check = (len(right) == 1 and left == right and right[0][2]
                              and right[0][1] in {"object", "str", "int"})
                if none_check or enum_check:
                    p_rel, ln_s, col_s = key[:-1].rsplit(":", 2)
                    self.profile_sites.setdefault(p_rel, set()).add((int(ln_s), int(col_s)))

    @lru_cache(maxsize=64)
    def original_lines(self, path: str):
        return self.archive.read(path).decode("utf-8").splitlines(keepends=True)

    @lru_cache(maxsize=64)
    def _original_profile_signature(self, path: str) -> str:
        tree = ast.parse(self.archive.read(path), type_comments=True)
        return _signature(tree, self.identity_classes, self.enum_fields, sites=self.profile_sites[path])

    def reconstruct(self, change: dict) -> bytes | None:
        path = change["old"]
        if path != change["new"] or not path.endswith(".py") or not change["hunks"]:
            return None
        try:
            lines = self.original_lines(path)
        except (KeyError, UnicodeError):
            return None
        output = []
        cursor = 0
        for start, old_count, new_count, hunk in change["hunks"]:
            offset = start - 1 if old_count else start
            old = [s[1:] for s in hunk if s[0] in " -"]
            new = [s[1:] for s in hunk if s[0] in " +"]
            if (offset < cursor or len(old) != old_count or len(new) != new_count
                    or old != [s.rstrip("\r\n") for s in lines[offset:offset + old_count]]):
                return None
            output.extend(lines[cursor:offset])
            output.extend(s + "\n" for s in new)
            cursor = offset + old_count
        output.extend(lines[cursor:])
        return "".join(output).encode("utf-8")

    def analyze(self, change: dict) -> str:
        raw = self.reconstruct(change)
        if raw is None:
            return "unknown"
        path = change["old"]
        try:
            tree = ast.parse(raw, type_comments=True)
            compile(tree, path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError, UnicodeError, RecursionError):
            return "invalid"
        if _has_undefined_import_time_name(tree):
            return "invalid"

        self._last_parsed = (path, raw, tree)
        before_cf = self._cf_hashes.get(path)
        if before_cf is not None and _control_flow_hash(tree) != before_cf:
            return "structural"

        edits = change["added"] + change["removed"]
        has_special = any(SPECIAL_COMMENT_RE.search(s) for s in edits)
        has_coding = any(CODING_RE.match(l) for l in raw.splitlines()[:2])
        if not has_special and not has_coding:
            has_eq_or_fmt = True
            if has_eq_or_fmt:
                before_sig = self._sig_hashes.get(path)
                if before_sig is not None and _signature(ast.parse(raw, type_comments=True), self.identity_classes, self.enum_fields) == before_sig:
                    return "equivalent"
                if any("None" in s or " is " in s or "==" in s or "!=" in s for s in edits):
                    sites = self.profile_sites.get(path)
                    if sites:
                        if _signature(ast.parse(raw, type_comments=True), self.identity_classes, self.enum_fields, sites=sites) == self._original_profile_signature(path):
                            return "equivalent"

        return "changed"

    def filter_unchanged_function_lines(self, change: dict) -> tuple[set[int], list[int]]:
        old_lines = set(change["old_lines"])
        anchors = list(change["insertion_anchors"])
        path = change["old"]
        if not path or path != change["new"] or not path.endswith(".py") or (not old_lines and not anchors):
            return old_lines, anchors
        if self._last_parsed is None or self._last_parsed[0] != path:
            raw = self.reconstruct(change)
            if raw is None:
                return old_lines, anchors
            try:
                after_tree = ast.parse(raw, type_comments=True)
            except SyntaxError:
                return old_lines, anchors
        else:
            after_tree = self._last_parsed[2]

        before_blocks = self._baseline_blocks_cache.get(path)
        if before_blocks is None:
            try:
                before_tree = ast.parse(self.archive.read(path), type_comments=True)
            except (KeyError, SyntaxError):
                return old_lines, anchors
            before_blocks = _function_block_hashes(before_tree)
            self._baseline_blocks_cache[path] = before_blocks

        after_blocks = _function_block_hashes(after_tree)
        unchanged = [
            (s, e)
            for name, (s, e, h) in before_blocks.items()
            if name in after_blocks and after_blocks[name][2] == h
        ]
        if not unchanged:
            return old_lines, anchors

        filtered_lines = {ln for ln in old_lines if not any(s <= ln <= e for s, e in unchanged)}
        filtered_anchors = [a for a in anchors if not any(s < a <= e for s, e in unchanged)]
        return filtered_lines, filtered_anchors

    def check_test_file(self, change: dict, sqlglot_root: Path) -> str:
        path = change["old"]
        if not path or path != change["new"] or not path.endswith(".py") or not change["hunks"]:
            return "unknown"
        fp = sqlglot_root / path
        if not fp.is_file():
            return "unknown"
        try:
            raw_before = fp.read_bytes()
            lines = raw_before.decode("utf-8").splitlines(keepends=True)
        except (OSError, UnicodeError):
            return "unknown"
        output = []
        cursor = 0
        for start, old_count, new_count, hunk in change["hunks"]:
            offset = start - 1 if old_count else start
            old = [s[1:] for s in hunk if s[0] in " -"]
            new = [s[1:] for s in hunk if s[0] in " +"]
            if (offset < cursor or len(old) != old_count or len(new) != new_count
                    or old != [s.rstrip("\r\n") for s in lines[offset:offset + old_count]]):
                return "unknown"
            output.extend(lines[cursor:offset])
            output.extend(s + "\n" for s in new)
            cursor = offset + old_count
        output.extend(lines[cursor:])
        raw_after = "".join(output).encode("utf-8")
        try:
            after_tree = ast.parse(raw_after, type_comments=True)
            compile(after_tree, path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError, UnicodeError, RecursionError):
            return "invalid"
        edits = change["added"] + change["removed"]
        if not any(SPECIAL_COMMENT_RE.search(s) for s in edits) and not any(
            CODING_RE.match(l) for l in raw_after.splitlines()[:2]
        ):
            try:
                before_tree = ast.parse(raw_before, type_comments=True)
                if ast.dump(before_tree, include_attributes=False) == ast.dump(after_tree, include_attributes=False):
                    return "equivalent"
            except SyntaxError:
                pass
        return "valid"

    def dict_value_changes(self, change: dict, tracked_attrs: set[str]) -> tuple[str, list[str]] | None:
        """If the ONLY semantic change in the file is modifying values of existing static keys
        inside one tracked class-level dict literal (no keys added/removed, no **spread edits),
        return (attr_name, sorted_changed_key_tokens). Otherwise return None."""
        path = change["old"]
        if not path or path != change["new"] or not path.endswith(".py") or not tracked_attrs:
            return None
        if self._last_parsed is None or self._last_parsed[0] != path:
            raw = self.reconstruct(change)
            if raw is None:
                return None
            try:
                after_tree = ast.parse(raw, type_comments=True)
            except SyntaxError:
                return None
        else:
            after_tree = self._last_parsed[2]
        try:
            before_tree = ast.parse(self.archive.read(path), type_comments=True)
        except (KeyError, SyntaxError):
            return None

        b_dicts = _extract_class_dicts(before_tree, tracked_attrs)
        a_dicts = _extract_class_dicts(after_tree, tracked_attrs)
        if not b_dicts or len(b_dicts) != len(a_dicts):
            return None

        # Find which dict(s) changed and verify nothing outside them changed
        changed_attr: str | None = None
        changed_keys: list[str] = []
        target_pair: tuple[ast.Dict, ast.Dict] | None = None
        for (bp, ba, bd), (ap, aa, ad) in zip(b_dicts, a_dicts):
            if bp != ap or ba != aa:
                return None
            if ast.dump(bd, include_attributes=False) == ast.dump(ad, include_attributes=False):
                continue
            if changed_attr is not None:
                return None  # multiple dicts edited -> fall back safely
            if len(bd.keys) != len(ad.keys):
                return None
            b_map: dict[str, str] = {}
            a_map: dict[str, str] = {}
            b_spreads: list[str] = []
            a_spreads: list[str] = []
            for k, v in zip(bd.keys, bd.values):
                if k is None:
                    b_spreads.append(ast.dump(v, include_attributes=False))
                else:
                    tok = _ast_key_token(k)
                    if tok is None or tok in b_map:
                        return None
                    b_map[tok] = ast.dump(v, include_attributes=False)
            for k, v in zip(ad.keys, ad.values):
                if k is None:
                    a_spreads.append(ast.dump(v, include_attributes=False))
                else:
                    tok = _ast_key_token(k)
                    if tok is None or tok in a_map:
                        return None
                    a_map[tok] = ast.dump(v, include_attributes=False)
            if b_spreads != a_spreads or set(b_map) != set(a_map):
                return None
            diff_k = sorted(k for k in b_map if b_map[k] != a_map[k])
            if not diff_k:
                return None
            b_val_nodes = {_ast_key_token(k): v for k, v in zip(bd.keys, bd.values) if k is not None}
            a_val_nodes = {_ast_key_token(k): v for k, v in zip(ad.keys, ad.values) if k is not None}
            if not all(_is_safe_dict_value(b_val_nodes.get(k), a_val_nodes.get(k)) for k in diff_k):
                return None
            changed_attr = ba
            changed_keys = diff_k
            target_pair = (bd, ad)

        if changed_attr is None or target_pair is None:
            return None
        bd, ad = target_pair
        bk, bv, ak, av = bd.keys, bd.values, ad.keys, ad.values
        try:
            bd.keys = []; bd.values = []
            ad.keys = []; ad.values = []
            if hasattr(before_tree, "type_ignores"): before_tree.type_ignores = []
            if hasattr(after_tree, "type_ignores"): after_tree.type_ignores = []
            same_rest = ast.dump(before_tree, include_attributes=False) == ast.dump(after_tree, include_attributes=False)
        finally:
            bd.keys, bd.values, ad.keys, ad.values = bk, bv, ak, av
        if not same_rest:
            return None
        return changed_attr, changed_keys
