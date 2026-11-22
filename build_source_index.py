"""Build a source archive and static analysis metadata from pinned sqlglot."""

from __future__ import annotations

import ast
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from source_analysis import _control_flow_hash, _signature


def _module_name(rel_posix: str) -> str:
    p = PurePosixPath(rel_posix)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


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


def _build_expr_symbols(root: Path) -> dict:
    """Map concrete Expression subclass ranges + body-only reference regions + Func metaclass subclasses."""
    expr_classes: dict[str, list[list]] = defaultdict(list)
    class_children: dict[str, set[str]] = defaultdict(set)
    all_expr_names: set[str] = set()
    expr_dir = root / "sqlglot" / "expressions"
    if not expr_dir.exists():
        return {"classes": {}, "refs": {}, "func_classes": []}

    for f in sorted(expr_dir.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        if rel.endswith("__init__.py"):
            continue
        try:
            tree = ast.parse(f.read_bytes())
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                expr_classes[rel].append([node.lineno, node.end_lineno or node.lineno, node.name])
                all_expr_names.add(node.name)
                for base in node.bases:
                    bname = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else None)
                    if bname:
                        class_children[bname].add(node.name)

    # Identify all Func subclasses (dynamically registered into Parser.FUNCTIONS by _Func metaclass)
    func_classes: set[str] = {"Func"}
    changed = True
    while changed:
        changed = False
        for parent, kids in list(class_children.items()):
            if parent in func_classes:
                for kid in kids:
                    if kid not in func_classes:
                        func_classes.add(kid)
                        changed = True

    direct_refs: dict[str, dict[str, list[list[int]]]] = defaultdict(lambda: defaultdict(list))
    for scan_dir in (root / "sqlglot", root / "tests"):
        if not scan_dir.exists():
            continue
        for f in sorted(scan_dir.rglob("*.py")):
            rel = f.relative_to(root).as_posix()
            try:
                tree = ast.parse(f.read_bytes())
            except SyntaxError:
                continue

            def _visit_scope(node: ast.AST, in_func_body: bool) -> None:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body_start = node.body[0].lineno if node.body else node.lineno + 1
                    for sub in ast.walk(node):
                        ln = getattr(sub, "lineno", None)
                        if ln is None or ln < body_start:
                            continue
                        if isinstance(sub, ast.Name) and sub.id in all_expr_names:
                            direct_refs[sub.id][rel].append([ln, ln])
                        elif isinstance(sub, ast.Attribute) and sub.attr in all_expr_names:
                            direct_refs[sub.attr][rel].append([ln, ln])
                    return
                for child in ast.iter_child_nodes(node):
                    _visit_scope(child, in_func_body)

            _visit_scope(tree, False)

    # Propagate references upward from subclasses to base classes
    merged_refs: dict[str, dict[str, list[list[int]]]] = {}
    for sym in all_expr_names:
        descendants = {sym}
        stack = [sym]
        while stack:
            cur = stack.pop()
            for kid in class_children.get(cur, ()):
                if kid not in descendants:
                    descendants.add(kid)
                    stack.append(kid)
        file_ranges: dict[str, list[list[int]]] = defaultdict(list)
        for d in descendants:
            for rf, regs in direct_refs.get(d, {}).items():
                file_ranges[rf].extend(regs)
        compact: dict[str, list[list[int]]] = {}
        for rf, regs in file_ranges.items():
            regs.sort()
            merged: list[list[int]] = []
            for s, e in regs:
                if merged and s <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            compact[rf] = merged
        merged_refs[sym] = compact

    return {"classes": dict(expr_classes), "refs": merged_refs, "func_classes": sorted(func_classes)}


def build(root: Path, destination: Path) -> None:
    names = Counter()
    plain = set()
    subclassed = set()
    conditions: dict[str, dict[int, int]] = {}
    stmt_starts: dict[str, dict[int, int]] = {}
    enum_bases: dict[str, list[str]] = defaultdict(list)
    enum_has_primitive_base: set[str] = set()
    values: dict[str, set[str]] = defaultdict(set)
    scalar_bindings: dict[str, dict[int, str]] = defaultdict(dict)
    read_sites: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    def_headers: dict[str, list[list[int]]] = defaultdict(list)
    tokenizer_ranges: dict[str, list[list[int]]] = defaultdict(list)
    test_methods: dict[str, list[list]] = defaultdict(list)
    test_importers: dict[str, set[str]] = defaultdict(set)
    parsed_sqlglot: list[tuple[str, bytes]] = []

    sqlglot_files = sorted((root / "sqlglot").rglob("*.py"))
    test_files = sorted((root / "tests").rglob("*.py")) if (root / "tests").exists() else []
    known_modules: dict[str, str] = {}
    for f in sqlglot_files + test_files:
        rel = f.relative_to(root).as_posix()
        mod = _module_name(rel)
        if mod:
            known_modules[mod] = rel

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sqlglot_files + test_files:
            raw = file.read_bytes()
            rel = file.relative_to(root).as_posix()
            try:
                tree = ast.parse(raw, type_comments=True)
            except SyntaxError:
                continue

            if rel.startswith("tests/"):
                # Track test-file imports so shared test bases/helpers are never method-narrowed
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in known_modules and known_modules[alias.name].startswith("tests/"):
                                test_importers[known_modules[alias.name]].add(rel)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        if node.module in known_modules and known_modules[node.module].startswith("tests/"):
                            test_importers[known_modules[node.module]].add(rel)
                if Path(rel).name.startswith("test_"):
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                            read_sites[node.attr][rel].add(node.lineno)
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef):
                            for item in node.body:
                                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_") and item.body:
                                    test_methods[rel].append([item.body[0].lineno, item.end_lineno or item.lineno, f"{rel}::{node.name}::{item.name}"])
                continue

            # sqlglot/ source file
            archive.writestr(rel, raw)
            parsed_sqlglot.append((rel, raw))
            module = ".".join(file.relative_to(root).with_suffix("").parts)

            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    read_sites[node.attr][rel].add(node.lineno)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                    first_body = node.body[0].lineno
                    if first_body > node.lineno + 1:
                        def_headers[rel].append([node.lineno, first_body - 1])
                elif isinstance(node, ast.ClassDef):
                    if rel.startswith("sqlglot/dialects/") and node.name == "Tokenizer":
                        tokenizer_ranges[rel].append([node.lineno, node.end_lineno or node.lineno])
                    for stmt in node.body:
                        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and _is_scalar_const(stmt.value):
                            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                            for target in targets:
                                if isinstance(target, ast.Name):
                                    for ln in range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1):
                                        scalar_bindings[rel][ln] = target.id
                    if not any(getattr(n, "name", "") in {"__eq__", "__ne__", "_missing_"} for n in node.body):
                        plain.add(f"{module}.{node.name}")
                    names[node.name] += 1
                    for base in node.bases:
                        bname = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else None)
                        if bname:
                            subclassed.add(bname)
                            enum_bases[node.name].append(bname)
                            if bname in {"str", "int", "StrEnum", "IntEnum", "IntFlag", "AutoName"}:
                                enum_has_primitive_base.add(node.name)
                elif isinstance(node, (ast.If, ast.While)):
                    test = node.test
                    if test.lineno != test.end_lineno:
                        file_conds = conditions.setdefault(rel, {})
                        for line in range(test.lineno + 1, (test.end_lineno or test.lineno) + 1):
                            file_conds[line] = test.lineno
                elif isinstance(node, (ast.Return, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Raise, ast.Assert)):
                    if node.end_lineno and node.end_lineno > node.lineno:
                        file_stmts = stmt_starts.setdefault(rel, {})
                        for line in range(node.lineno + 1, node.end_lineno + 1):
                            file_stmts.setdefault(line, node.lineno)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                            val = node.value
                            if isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name) and val.attr.isupper():
                                values[target.attr].add(val.value.id)
                            elif isinstance(val, ast.Name):
                                values[target.attr].add(f"@{val.id}")
                            else:
                                values[target.attr].add("?")
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Attribute) and isinstance(node.target.value, ast.Name) and node.target.value.id == "self":
                        if isinstance(node.annotation, ast.Name):
                            values[node.target.attr].add(node.annotation.id)

        # Safe identity classes (no custom __eq__/__ne__ and not subclassed)
        safe = sorted(full for full in plain if names[full.rsplit(".", 1)[-1]] == 1 and full.rsplit(".", 1)[-1] not in subclassed)

        # Safe Enum classes ONLY: exclude str/int/StrEnum/IntEnum/AutoName subclasses where == != is!
        enums = {"Enum"}
        changed = True
        while changed:
            changed = False
            for cls_name, bases in enum_bases.items():
                if cls_name in enum_has_primitive_base:
                    continue
                if any(b in enum_has_primitive_base for b in bases):
                    enum_has_primitive_base.add(cls_name)
                    continue
                if cls_name not in enums and any(b in enums for b in bases):
                    enums.add(cls_name)
                    changed = True
        enum_fields = {}
        for name, tags in values.items():
            types = set()
            for tag in tags:
                types.update(values.get(tag[1:], {"?"}) if tag.startswith("@") else {tag})
            if len(types) == 1 and types <= enums:
                enum_fields[name] = types.pop()

        safe_set = set(safe)
        cf_hashes = {}
        sig_hashes = {}
        for rel, raw in parsed_sqlglot:
            t_exact = ast.parse(raw, type_comments=True)
            cf_hashes[rel] = _control_flow_hash(t_exact)
            sig_hashes[rel] = _signature(t_exact, safe_set, enum_fields)

        fields = {name for bindings in scalar_bindings.values() for name in bindings.values()}
        archive.writestr("field-coverage.json", json.dumps([
            scalar_bindings,
            {name: {p: sorted(lines) for p, lines in read_sites[name].items()} for name in sorted(fields)},
        ]))
        archive.writestr("identity-classes.json", json.dumps(safe))
        archive.writestr("condition-starts.json", json.dumps(conditions))
        archive.writestr("stmt-starts.json", json.dumps(stmt_starts))
        archive.writestr("enum-fields.json", json.dumps(enum_fields))
        archive.writestr("expr-symbols.json", json.dumps(_build_expr_symbols(root)))
        archive.writestr("def-headers.json", json.dumps({k: sorted(v) for k, v in def_headers.items()}))
        archive.writestr("tokenizer-ranges.json", json.dumps({k: sorted(v) for k, v in tokenizer_ranges.items()}))
        archive.writestr("test-methods.json", json.dumps({k: sorted(v) for k, v in test_methods.items()}))
        archive.writestr("test-importers.json", json.dumps({k: sorted(v) for k, v in test_importers.items()}))
        archive.writestr("baseline-hashes.json", json.dumps({"cf": cf_hashes, "sig": sig_hashes}))
        if (root / "LICENSE").exists():
            archive.write(root / "LICENSE", "SQLGlot-LICENSE")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
