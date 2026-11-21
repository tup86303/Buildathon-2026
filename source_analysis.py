"""Compare patched Python to its baseline without executing any patch."""

import ast
import json
import zipfile
from functools import lru_cache


def _annotation_names(annotation):
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _annotation_names(annotation.left) | _annotation_names(annotation.right)
    if isinstance(annotation, ast.Name):
        return {annotation.id}
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return set()
    return {"?"}


def _signature(tree, identity_classes, enum_fields):
    identity_classes = {alias.asname or alias.name for node in ast.walk(tree)
                        if isinstance(node, ast.ImportFrom) for alias in node.names
                        if f"{node.module}.{alias.name}" in identity_classes}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        init = next((m for m in cls.body if isinstance(m, ast.FunctionDef)
                     and m.name == "__init__"), None)
        if init is None:
            continue
        fields = {n.target.attr for n in ast.walk(init) if isinstance(n, ast.AnnAssign)
                  and isinstance(n.target, ast.Attribute) and isinstance(n.target.value, ast.Name)
                  and n.target.value.id == "self" and _annotation_names(n.annotation) <= identity_classes}
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            for cmp in ast.walk(method):
                if not isinstance(cmp, ast.Compare) or len(cmp.ops) != 1:
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
    return ast.dump(tree, include_attributes=False)


class SourceIndex:
    def __init__(self, archive):
        self.archive = zipfile.ZipFile(archive)
        self.identity_classes = set(json.loads(self.archive.read("identity-classes.json")))
        self.conditions = json.loads(self.archive.read("condition-starts.json"))
        self.enum_fields = json.loads(self.archive.read("enum-fields.json"))

    @lru_cache(maxsize=32)
    def original(self, path):
        text = self.archive.read(path).decode("utf-8").splitlines(keepends=True)
        return text, _signature(ast.parse("".join(text), type_comments=True), self.identity_classes, self.enum_fields)

    def equivalent(self, change):
        path = change["old"]
        if path != change["new"] or not path.endswith(".py") or not change["hunks"]:
            return False
        # An inexpensive prefilter. The complete module comparison makes the decision.
        edits = change["added"] + change["removed"]
        if any(s.strip() and not s.lstrip().startswith("#") for s in edits):
            if not any("None" in s or " is " in s for s in edits):
                return False
        try:
            lines, before = self.original(path)
            output = []
            cursor = 0
            for start, old_count, new_count, hunk in change["hunks"]:
                offset = start - 1 if old_count else start
                old = [s[1:] for s in hunk if s[0] in " -"]
                new = [s[1:] for s in hunk if s[0] in " +"]
                if (offset < cursor or len(old) != old_count or len(new) != new_count
                        or old != [s.rstrip("\r\n") for s in lines[offset:offset+old_count]]):
                    return False
                output.extend(lines[cursor:offset])
                output.extend(s + "\n" for s in new)
                cursor = offset + old_count
            output.extend(lines[cursor:])
            tree = ast.parse("".join(output), type_comments=True)
            return _signature(tree, self.identity_classes, self.enum_fields) == before
        except (KeyError, SyntaxError, UnicodeError, ValueError, RecursionError):
            return False
