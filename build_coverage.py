"""Convert pytest-cov's per-test contexts into a compact runtime lookup map."""

from __future__ import annotations

import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def _relative(path: str, root: Path) -> str | None:
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def _scopes(root: Path) -> dict[str, list[list[int]]]:
    result: dict[str, list[list[int]]] = {}
    for path in (root / "sqlglot").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        spans: list[list[int]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                starts = [node.lineno, *(item.lineno for item in node.decorator_list)]
                spans.append([min(starts), node.end_lineno or node.lineno])
        relative = _relative(str(path), root)
        if relative:
            result[relative] = spans
    return result


def _coverage(path: Path, root: Path) -> dict[str, dict[str, list[int]]]:
    from coverage import Coverage

    coverage = Coverage(data_file=str(path))
    coverage.load()
    data = coverage.get_data()
    result: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for measured_file in data.measured_files():
        relative = _relative(measured_file, root)
        if not relative or not relative.startswith("sqlglot/"):
            continue
        for line, contexts in data.contexts_by_lineno(measured_file).items():
            for context in contexts:
                if context.endswith("|run"):
                    test_id = context[:-4]
                    result[relative][test_id].add(line)
    return {
        file: {test_id: sorted(lines) for test_id, lines in tests.items()}
        for file, tests in result.items()
    }


def _access(path: Path, root: Path) -> dict[str, list[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    access_files = list(path.parent.glob(path.name + ".*"))
    if path.exists():
        access_files.append(path)
    for access_file in access_files:
        for line in access_file.read_text(encoding="utf-8").splitlines():
            relative, separator, test_id = line.partition("\t")
            if separator and relative and test_id:
                result[relative].add(test_id)
    return {file: sorted(test_ids) for file, test_ids in result.items()}


def main() -> None:
    coverage_file = Path(sys.argv[1])
    access_file = Path(sys.argv[2])
    root = Path(sys.argv[3]).resolve()
    output = Path(sys.argv[4])
    result = {
        "source": _coverage(coverage_file, root),
        "access": _access(access_file, root),
        "scopes": _scopes(root),
    }
    output.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    os.chmod(output, 0o644)
    print(
        f"coverage map: {len(result['source'])} source files, "
        f"{sum(map(len, result['source'].values()))} tests",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
