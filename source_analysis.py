"""Compare patched Python to its baseline without executing any patch."""

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
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Return,
    ast.Raise,
    ast.Yield,
    ast.YieldFrom,
    ast.Break,
    ast.Continue,
    ast.Assert,
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


def _signature_hash(tree, identity_classes, enum_fields, infer_eq=True, sites=None) -> str:
    if infer_eq:
        id_cls = {alias.asname or alias.name for node in ast.walk(tree)
                  if isinstance(node, ast.ImportFrom) for alias in node.names
                  if f"{node.module}.{alias.name}" in identity_classes}
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            init = next((m for m in cls.body if isinstance(m, ast.FunctionDef)
                         and m.name == "__init__"), None)
            if init is None:
                continue
            fields = {n.target.attr for n in ast.walk(init) if isinstance(n, ast.AnnAssign)
                      and isinstance(n.target, ast.Attribute) and isinstance(n.target.value, ast.Name)
                      and n.target.value.id == "self" and _annotation_names(n.annotation) <= id_cls}
            for method in cls.body:
                if not isinstance(method, ast.FunctionDef):
                    continue
                for cmp in ast.walk(method):
                    if not isinstance(cmp, ast.Compare) or len(cmp.ops) != 1:
                        continue
                    if sites is not None and (cmp.lineno, cmp.col_offset) not in sites:
                        continue
                    left, right = cmp.left, cmp.comparators[0]
                    if not (isinstance(left, ast.Attribute) and isinstance(left.value, ast.Name)
                            and left.value.id == "self"):
                        continue
                    plain_none = left.attr in fields and isinstance(right, ast.Constant) and right.value is None
                    enum_member = (isinstance(right, ast.Attribute) and right.attr.isupper()
                                   and isinstance(right.value, ast.Name)
                                   and enum_fields.get(left.attr) == right.value.id)
                    if not (plain_none or enum_member):
                        continue
                    if isinstance(cmp.ops[0], ast.Eq):
                        cmp.ops[0] = ast.Is()
                    elif isinstance(cmp.ops[0], ast.NotEq):
                        cmp.ops[0] = ast.IsNot()
    return hashlib.blake2b(ast.dump(tree, include_attributes=False).encode("utf-8"), digest_size=16).hexdigest()


def _control_flow_hash(tree) -> str:
    items = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append((
                type(node).__name__,
                node.name,
                len(node.args.posonlyargs),
                len(node.args.args),
                len(node.args.kwonlyargs),
                len(node.args.defaults),
                len(node.decorator_list),
            ))
        elif isinstance(node, CONTROL_FLOW_NODES):
            items.append((type(node).__name__,))
    return hashlib.blake2b(repr(items).encode("utf-8"), digest_size=16).hexdigest()



def _function_block_hashes(tree: ast.AST, identity_classes, enum_fields, infer_eq: bool) -> dict[str, tuple[int, int, str]]:
    """Compute testmon-style normalized AST block hashes per FunctionDef/AsyncFunctionDef.

    Docstrings and type annotations are stripped so comments, formatting, line-number
    shifts, and annotation-only edits inside an otherwise untouched function produce
    the exact same block hash.
    """
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


class SourceIndex:
    def __init__(self, archive, infer_eq: bool | None = None):
        self.archive = zipfile.ZipFile(archive)
        names = set(self.archive.namelist())
        self.subprocess_readers = json.loads(self.archive.read("subprocess-readers.json")) if "subprocess-readers.json" in names else {}
        self.scalar_bindings, self.read_sites = json.loads(self.archive.read("field-coverage.json")) if "field-coverage.json" in names else ({}, {})
        self.doctest_suites = json.loads(self.archive.read("doctest-suites.json")) if "doctest-suites.json" in names else []
        self.def_headers = json.loads(self.archive.read("def-headers.json")) if "def-headers.json" in names else {}
        self.test_methods = json.loads(self.archive.read("test-methods.json")) if "test-methods.json" in names else {}
        self.identity_classes = set(json.loads(self.archive.read("identity-classes.json")))
        self.conditions = json.loads(self.archive.read("condition-starts.json"))
        self.enum_fields = json.loads(self.archive.read("enum-fields.json"))
        self.module_dependents = json.loads(self.archive.read("module-dependents.json")) if "module-dependents.json" in names else {}
        self.expr_symbols = json.loads(self.archive.read("expr-symbols.json")) if "expr-symbols.json" in names else {}
        dp_file = Path(archive).with_name("dialect-profile.json.gz")
        if dp_file.exists():
            import gzip as _gzip
            self.dialect_profile = json.loads(_gzip.decompress(dp_file.read_bytes()))
        elif "dialect-profile.json" in names:
            self.dialect_profile = json.loads(self.archive.read("dialect-profile.json"))
        else:
            self.dialect_profile = {}
        if "baseline-hashes.json" in names:
            hashes = json.loads(self.archive.read("baseline-hashes.json"))
            self._cf_hashes = hashes.get("cf", {})
            self._sig_eq = hashes.get("sig_eq", {})
            self._sig_exact = hashes.get("sig_exact", {})
        else:
            self._cf_hashes = {}
            self._sig_eq = {}
            self._sig_exact = {}
        self.infer_eq = False
        self._last_parsed = None
        self._baseline_blocks_cache = {}
        self.profile_sites = {}
        profile = Path(archive).with_name("operand-profile.json.gz")
        if profile.is_file() and infer_eq is not False and os.environ.get("SELECTOR_ENABLE_EQ_EQUIV") != "0":
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
                    path, line, col = key[:-1].rsplit(":", 2)
                    self.profile_sites.setdefault(path, set()).add((int(line), int(col)))

    @lru_cache(maxsize=64)
    def original_lines(self, path):
        raw = self.archive.read(path)
        return raw.decode("utf-8").splitlines(keepends=True)

    def _baseline_hashes(self, path):
        cf = self._cf_hashes.get(path)
        sig_map = self._sig_eq if self.infer_eq else self._sig_exact
        sig = sig_map.get(path)
        if cf is not None and sig is not None:
            return cf, sig
        raw = self.archive.read(path)
        tree = ast.parse(raw, type_comments=True)
        cf = _control_flow_hash(tree)
        sig = _signature_hash(tree, self.identity_classes, self.enum_fields, self.infer_eq)
        self._cf_hashes[path] = cf
        sig_map[path] = sig
        return cf, sig

    def _reconstruct(self, change):
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

    def analyze(self, change) -> str:
        """Classify a Python patch as equivalent, syntax_error, structural, local_expr, or unknown."""
        raw_after = self._reconstruct(change)
        if raw_after is None:
            return "unknown"
        path = change["old"]

        first_two = raw_after.splitlines()[:2]
        has_coding_cookie = any(CODING_RE.match(line) for line in first_two)

        try:
            tree = ast.parse(raw_after, type_comments=True)
            compile(tree, path, "exec", dont_inherit=True)
        except (SyntaxError, UnicodeError, ValueError, RecursionError):
            return "syntax_error"

        self._last_parsed = (path, raw_after, tree)
        before_cf, before_sig = self._baseline_hashes(path)

        edits = change["added"] + change["removed"]
        has_special_comment = any(SPECIAL_COMMENT_RE.search(s) for s in edits)
        maybe_equiv = not has_coding_cookie and not has_special_comment
        if maybe_equiv:
            if _signature_hash(tree, self.identity_classes, self.enum_fields, self.infer_eq) == before_sig:
                return "equivalent"
            sites = self.profile_sites.get(path)
            if sites and change["added"] and any(" is " in s for s in edits):
                before = ast.parse(self.archive.read(path), type_comments=True)
                if (_signature_hash(tree, self.identity_classes, self.enum_fields, True, sites)
                        == _signature_hash(before, self.identity_classes, self.enum_fields, True, sites)):
                    return "equivalent"

        if _control_flow_hash(tree) != before_cf:
            return "structural"
        return "local_expr"

    def equivalent(self, change):
        return self.analyze(change) == "equivalent"

    def filter_unchanged_function_lines(self, change: dict) -> tuple[set[int], list[int]]:
        """Drop changed lines and insertion anchors inside functions whose normalized AST block hash is unchanged."""
        old_lines = set(change["old_lines"])
        anchors = list(change["insertion_anchors"])
        path = change["old"]
        if not path or path != change["new"] or not path.endswith(".py"):
            return old_lines, anchors
        if not old_lines and not anchors:
            return old_lines, anchors
        if self._last_parsed is None or self._last_parsed[0] != path:
            raw_after = self._reconstruct(change)
            if raw_after is None:
                return old_lines, anchors
            try:
                after_tree = ast.parse(raw_after, type_comments=True)
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
            before_blocks = _function_block_hashes(before_tree, self.identity_classes, self.enum_fields, self.infer_eq)
            self._baseline_blocks_cache[path] = before_blocks

        after_blocks = _function_block_hashes(after_tree, self.identity_classes, self.enum_fields, self.infer_eq)
        unchanged_ranges = [
            (s, e)
            for name, (s, e, h) in before_blocks.items()
            if name in after_blocks and after_blocks[name][2] == h
        ]
        if not unchanged_ranges:
            return old_lines, anchors

        def _in_unchanged(ln: int, strict_start: bool = False) -> bool:
            return any((s < ln <= e) if strict_start else (s <= ln <= e) for s, e in unchanged_ranges)

        filtered_lines = {ln for ln in old_lines if not _in_unchanged(ln, strict_start=False)}
        filtered_anchors = [a for a in anchors if not _in_unchanged(a, strict_start=True)]
        return filtered_lines, filtered_anchors

    def check_test_file(self, change: dict, sqlglot_root: Path) -> str:
        """Validate a patched tests/*.py file against sqlglot_root: returns 'syntax_error', 'equivalent', 'valid', or 'unknown'."""
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
        except (SyntaxError, UnicodeError, ValueError, RecursionError):
            return "syntax_error"
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
