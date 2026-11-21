"""Pack the pinned Python sources and a conservative class-equality index."""

import ast
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def build(root, destination):
    names = Counter()
    plain = set()
    subclassed = set()
    conditions = {}
    enum_bases = defaultdict(list)
    values = defaultdict(set)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted((root / "sqlglot").rglob("*.py")):
            raw = file.read_bytes()
            archive.writestr(file.relative_to(root).as_posix(), raw)
            module = ".".join(file.relative_to(root).with_suffix("").parts)
            tree = ast.parse(raw)
            for stmt in ast.walk(tree):
                if isinstance(stmt, ast.ClassDef):
                    if not any(getattr(n, "name", "") in {"__eq__", "__ne__", "_missing_"}
                               for n in stmt.body):
                        enum_bases[stmt.name].append({ast.unparse(b) for b in stmt.bases})
                if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
                    continue
                rhs = stmt.value
                tag = "?"
                if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name):
                    tag = rhs.func.id
                if isinstance(rhs, ast.Attribute) and isinstance(rhs.value, ast.Name):
                    tag = ("@" + rhs.attr if rhs.value.id == "self" else rhs.value.id)
                for target in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]):
                    if isinstance(target, ast.Attribute):
                        values[target.attr].add(tag)
                    elif isinstance(target, ast.Name) and target.id.isupper():
                        values[target.id].add(tag)
            starts = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.BoolOp, ast.Compare, ast.IfExp)) and node.end_lineno > node.lineno:
                    for line in range(node.lineno + 1, node.end_lineno + 1):
                        starts[line] = min(starts.get(line, node.lineno), node.lineno)
            if starts:
                conditions[file.relative_to(root).as_posix()] = starts
            aliases = {alias.asname or alias.name: f"{node.module}.{alias.name}"
                       for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                       for alias in node.names}
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                qualified = f"{module}.{node.name}"
                names[qualified] += 1
                for base in node.bases:
                    subclassed.update(aliases.get(n.id, f"{module}.{n.id}")
                                      for n in ast.walk(base) if isinstance(n, ast.Name))
                    subclassed.update(n.attr for n in ast.walk(base) if isinstance(n, ast.Attribute))
                members = {getattr(n, "name", "") for n in node.body}
                bindings = {n.id for stmt in node.body for n in ast.walk(stmt)
                            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                if not (node.bases or node.keywords or node.decorator_list
                        or {"__eq__", "__ne__"} & (members | bindings)):
                    plain.add(qualified)
        safe = sorted(name for name in plain - subclassed if names[name] == 1)
        archive.writestr("identity-classes.json", json.dumps(safe))
        archive.writestr("condition-starts.json", json.dumps(conditions))
        enums = {"Enum", "enum.Enum", "IntEnum", "StrEnum"}
        while True:
            more = {name for name, defs in enum_bases.items() if len(defs) == 1
                    and defs[0] & enums and defs[0] <= enums | {"str", "int"}}
            if more <= enums:
                break
            enums.update(more)
        enum_fields = {}
        for name, tags in values.items():
            types = set()
            for tag in tags:
                types.update(values.get(tag[1:], {"?"}) if tag.startswith("@") else {tag})
            if len(types) == 1 and types <= enums:
                enum_fields[name] = types.pop()
        archive.writestr("enum-fields.json", json.dumps(enum_fields))
        archive.write(root / "LICENSE", "SQLGlot-LICENSE")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
