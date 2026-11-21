"""Compare patched Python to its baseline without executing any patch."""

import ast
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


class SourceIndex:
    def __init__(self, archive, infer_eq: bool | None = None):
        self.archive = zipfile.ZipFile(archive)
        names = set(self.archive.namelist())
        self.subprocess_readers = json.loads(self.archive.read("subprocess-readers.json")) if "subprocess-readers.json" in names else {}
        self.scalar_bindings, self.read_sites = json.loads(self.archive.read("field-coverage.json")) if "field-coverage.json" in names else ({}, {})
        self.doctest_suites = json.loads(self.archive.read("doctest-suites.json")) if "doctest-suites.json" in names else []
        self.identity_classes = set(json.loads(self.archive.read("identity-classes.json")))
        self.conditions = json.loads(self.archive.read("condition-starts.json"))
        self.enum_fields = json.loads(self.archive.read("enum-fields.json"))
        self.module_dependents = json.loads(self.archive.read("module-dependents.json")) if "module-dependents.json" in names else {}
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
