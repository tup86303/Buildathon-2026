"""Pack the pinned Python sources, statement spans, precomputed AST hashes, and static dependency maps."""

import ast
import json
import sys
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

from source_analysis import _control_flow_hash, _signature_hash

MULTILINE_EXPR_NODES = (
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.BinOp,
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Subscript,
    ast.Return,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Raise,
    ast.Assert,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Attribute,
)

def _module_name(rel_posix: str) -> str:
    parts = list(Path(rel_posix).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_imports(rel_posix: str, tree: ast.AST, known_modules: dict[str, str]) -> set[str]:
    pkg_parts = list(Path(rel_posix).parent.parts)
    deps: set[str] = set()

    def _add_candidate(mod: str) -> None:
        if not mod:
            return
        if mod in known_modules:
            deps.add(known_modules[mod])
            return
        parts = mod.split(".")
        while len(parts) > 1:
            parts.pop()
            prefix = ".".join(parts)
            if prefix in known_modules:
                deps.add(known_modules[prefix])
                return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _add_candidate(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base_parts: list[str] = []
            if node.level:
                up = node.level - 1
                if up <= len(pkg_parts):
                    base_parts = pkg_parts[: len(pkg_parts) - up]
            if node.module:
                base_parts = base_parts + node.module.split(".")
            base = ".".join(base_parts)
            _add_candidate(base)
            for alias in node.names:
                if alias.name != "*":
                    _add_candidate(f"{base}.{alias.name}" if base else alias.name)
    deps.discard(rel_posix)
    return deps


def _statement_starts(tree: ast.AST) -> dict[int, list[int]]:
    starts: dict[int, set[int]] = defaultdict(set)
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        if lineno is None or end_lineno is None or end_lineno <= lineno:
            continue
        if isinstance(node, MULTILINE_EXPR_NODES):
            for line in range(lineno + 1, end_lineno + 1):
                starts[line].add(lineno)
        elif isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
            body = getattr(node, "body", None)
            header_end = (body[0].lineno - 1) if body else lineno
            if header_end > lineno:
                for line in range(lineno + 1, header_end + 1):
                    starts[line].add(lineno)
    return {line: sorted(vals) for line, vals in starts.items()}


def _python_child_imports(tree, path, modules):
    """Read imports from literal Python -c programs invoked by pinned tests."""
    for cls in [None, *(n for n in tree.body if isinstance(n, ast.ClassDef))]:
        for fn in (cls or tree).body:
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test_"):
                continue
            nodes = list(ast.walk(fn))
            strings = {t.id: n.value for n in nodes if isinstance(n, ast.Assign)
                       for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
            for n in nodes:
                if not isinstance(n, (ast.List, ast.Tuple)) or len(n.elts) < 3:
                    continue
                flag, code = n.elts[1:3]
                if not isinstance(flag, ast.Constant) or flag.value != "-c":
                    continue
                if isinstance(code, ast.Name):
                    code = strings.get(code.id)
                if not isinstance(code, ast.Constant) or not isinstance(code.value, str):
                    continue
                try:
                    child = ast.parse(code.value)
                except SyntaxError:
                    continue
                test = "::".join([path, *([cls.name] if cls else []), fn.name])
                yield test, _resolve_imports(path, child, modules)


def build(root, destination):
    names = Counter()
    plain = set()
    subclassed = set()
    conditions = {}
    enum_bases = defaultdict(list)
    values = defaultdict(set)
    parsed_sqlglot: list[tuple[str, bytes]] = []

    sqlglot_files = sorted((root / "sqlglot").rglob("*.py"))
    test_files = sorted((root / "tests").rglob("*.py")) if (root / "tests").exists() else []
    all_files = sqlglot_files + test_files

    known_modules: dict[str, str] = {}
    for file in all_files:
        rel = file.relative_to(root).as_posix()
        mod = _module_name(rel)
        if mod:
            known_modules[mod] = rel

    test_importers: dict[str, set[str]] = defaultdict(set)
    direct_sqlglot_importers: dict[str, set[str]] = defaultdict(set)
    direct_test_importers: dict[str, set[str]] = defaultdict(set)
    source_imports = {}
    child_imports = []
    scalar_bindings = defaultdict(dict)
    read_sites = defaultdict(lambda: defaultdict(set))
    doctest_suites = set()

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in all_files:
            raw = file.read_bytes()
            rel = file.relative_to(root).as_posix()
            try:
                tree = ast.parse(raw, type_comments=True)
            except SyntaxError:
                continue

            deps = _resolve_imports(rel, tree, known_modules)
            if rel.startswith("sqlglot/"):
                source_imports[rel] = deps
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                        read_sites[node.attr][rel].add(node.lineno)
                    elif isinstance(node, ast.ClassDef):
                        for stmt in node.body:
                            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or not isinstance(stmt.value, ast.Constant):
                                continue
                            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                            for target in targets:
                                if isinstance(target, ast.Name):
                                    for line in range(stmt.lineno, stmt.end_lineno + 1):
                                        scalar_bindings[rel][line] = target.id
            elif Path(rel).name.startswith("test_"):
                child_imports.extend(_python_child_imports(tree, rel, known_modules))
                if any(isinstance(n, ast.Attribute) and n.attr == "DocTestSuite" for n in ast.walk(tree)):
                    doctest_suites.add(rel)
            if rel.startswith("tests/"):
                for target in deps:
                    if target.startswith("tests/"):
                        test_importers[target].add(rel)
                    elif target.startswith("sqlglot/") and Path(rel).name.startswith("test_"):
                        direct_test_importers[target].add(rel)
            elif rel.startswith("sqlglot/") and Path(rel).name != "__init__.py":
                for target in deps:
                    if target.startswith("sqlglot/") and Path(target).name != "__init__.py":
                        direct_sqlglot_importers[target].add(rel)

            if not rel.startswith("sqlglot/"):
                continue

            archive.writestr(rel, raw)
            parsed_sqlglot.append((rel, raw))
            module = ".".join(file.relative_to(root).with_suffix("").parts)
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
            starts = _statement_starts(tree)
            if starts:
                conditions[rel] = starts
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

        module_dependents: dict[str, dict[str, list[str]]] = {}
        for rel in sorted(test_importers):
            visited: set[str] = set()
            queue = deque(test_importers[rel])
            while queue:
                dep = queue.popleft()
                if dep in visited:
                    continue
                visited.add(dep)
                for nxt in test_importers.get(dep, ()):
                    if nxt not in visited:
                        queue.append(nxt)
            test_deps = sorted(p for p in visited if Path(p).name.startswith("test_"))
            if test_deps:
                module_dependents[rel] = {"sqlglot": [], "tests": test_deps}

        for rel in sorted(set(direct_sqlglot_importers) | set(direct_test_importers)):
            sub_sqlglot: set[str] = set()
            if rel.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/")):
                queue = deque(
                    d for d in direct_sqlglot_importers.get(rel, ())
                    if d.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/"))
                )
                while queue:
                    dep = queue.popleft()
                    if dep in sub_sqlglot:
                        continue
                    sub_sqlglot.add(dep)
                    for nxt in direct_sqlglot_importers.get(dep, ()):
                        if nxt not in sub_sqlglot and nxt.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/")):
                            queue.append(nxt)
            test_deps = set(direct_test_importers.get(rel, ()))
            for s in sub_sqlglot:
                test_deps.update(direct_test_importers.get(s, ()))
            if sub_sqlglot or test_deps:
                module_dependents[rel] = {
                    "sqlglot": sorted(sub_sqlglot),
                    "tests": sorted(test_deps),
                }

        subclassed_short = {s.rsplit(".", 1)[-1] for s in subclassed}
        safe = sorted(
            name for name in plain - subclassed
            if names[name] == 1 and name.rsplit(".", 1)[-1] not in subclassed_short
        )
        safe_set = set(safe)
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

        # Precompute baseline hashes at build time so runtime never calls ast.parse on baseline!
        cf_hashes = {}
        sig_hashes_eq = {}
        sig_hashes_exact = {}
        for rel, raw in parsed_sqlglot:
            t_exact = ast.parse(raw, type_comments=True)
            cf_hashes[rel] = _control_flow_hash(t_exact)
            sig_hashes_exact[rel] = _signature_hash(t_exact, safe_set, enum_fields, infer_eq=False)
            sig_hashes_eq[rel] = _signature_hash(t_exact, safe_set, enum_fields, infer_eq=True)

        readers = defaultdict(set)
        for test, imported in child_imports:
            pending = list(imported)
            while pending:
                module = pending.pop()
                if test not in readers[module]:
                    readers[module].add(test)
                    pending.extend(source_imports.get(module, ()))
        archive.writestr("subprocess-readers.json", json.dumps({k: sorted(v) for k, v in readers.items()}))
        fields = {name for bindings in scalar_bindings.values() for name in bindings.values()}
        archive.writestr("field-coverage.json", json.dumps([scalar_bindings, {
            name: {p: sorted(lines) for p, lines in read_sites[name].items()} for name in sorted(fields)
        }]))
        archive.writestr("doctest-suites.json", json.dumps(sorted(doctest_suites)))
        archive.writestr("identity-classes.json", json.dumps(safe))
        archive.writestr("condition-starts.json", json.dumps(conditions))
        archive.writestr("module-dependents.json", json.dumps(module_dependents))
        archive.writestr("enum-fields.json", json.dumps(enum_fields))
        archive.writestr("baseline-hashes.json", json.dumps({
            "cf": cf_hashes,
            "sig_eq": sig_hashes_eq,
            "sig_exact": sig_hashes_exact,
        }))
        if (root / "LICENSE").exists():
            archive.write(root / "LICENSE", "SQLGlot-LICENSE")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
