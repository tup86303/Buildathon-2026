"""Select tests by mapping changed source regions to tests that execute them."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path, PurePosixPath

DATA = Path("/data")
APP = Path("/app")
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _patch_changes(patch: Path) -> list[dict]:
    changes: list[dict] = []
    current: dict | None = None
    old_line = 0
    new_line = 0
    for line in patch.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            old_path = parts[2][2:] if len(parts) > 2 and parts[2].startswith("a/") else ""
            new_path = parts[3][2:] if len(parts) > 3 and parts[3].startswith("b/") else ""
            current = {
                "old": "" if old_path == "/dev/null" else old_path,
                "new": "" if new_path == "/dev/null" else new_path,
                "old_lines": set(),
                "new_lines": set(),
                "added": [],
                "removed": [],
            }
            changes.append(current)
            continue
        if current is None:
            continue
        match = HUNK.match(line)
        if match:
            old_line = int(match.group(1))
            new_line = int(match.group(3))
            continue
        if line.startswith("\\"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            current["new_lines"].add(new_line)
            current["added"].append(line[1:])
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            current["old_lines"].add(old_line)
            current["removed"].append(line[1:])
            old_line += 1
        elif line.startswith(" "):
            old_line += 1
            new_line += 1
    return changes


def _normalized(path: str) -> str:
    return str(PurePosixPath(path))


def _tests_covering_region(source: dict, path: str, regions: list[tuple[int, int]]) -> set[str]:
    merged: list[list[int]] = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    result: set[str] = set()
    for test_id, lines in source.get(path, {}).items():
        region_index = 0
        for line in lines:
            while region_index < len(merged) and merged[region_index][1] < line:
                region_index += 1
            if region_index == len(merged):
                break
            if merged[region_index][0] <= line:
                result.add(test_id)
                break
    return result


def _module_tests(path: str, test_ids: list[str]) -> set[str]:
    path = _normalized(path)
    if path.startswith("tests/"):
        return {test_id for test_id in test_ids if test_id.startswith(path + "::")}

    stem = Path(path).stem
    candidates = {
        f"tests/test_{stem}",
        f"tests/{Path(path).parent.name}/test_{stem}",
    }
    return {
        test_id for test_id in test_ids
        if any(test_id.startswith(candidate + ".py::") for candidate in candidates)
    }


def _fixture_tests(path: str, access: dict, test_ids: list[str]) -> set[str]:
    path = _normalized(path)
    result = set(access.get(path, []))
    if result or not path.startswith("tests/fixtures/"):
        return result

    parts = PurePosixPath(path).parts
    if len(parts) > 3:
        category = parts[2]
        result.update(
            test_id for test_id in test_ids
            if test_id.startswith(f"tests/{category}/")
            or test_id.startswith(f"tests/test_{category}.py::")
        )
    return result


def _related_tests(path: str, test_ids: list[str]) -> set[str]:
    path = _normalized(path)
    if not path.startswith("sqlglot/"):
        return set()

    related: set[str] = set()
    if (
        path.startswith(("sqlglot/optimizer/", "sqlglot/parsers/", "sqlglot/dialects/"))
        or path.startswith("sqlglot/typing/")
        or path == "sqlglot/serde.py"
    ):
        related.update(
            test_id for test_id in test_ids
            if test_id.startswith("tests/test_optimizer.py::")
            or test_id.startswith("tests/optimizer/")
        )

    for package in ("dialects", "parsers"):
        prefix = f"sqlglot/{package}/"
        if path.startswith(prefix):
            stem = Path(path).stem
            related.update(
                test_id for test_id in test_ids
                if test_id.startswith(f"tests/dialects/test_{stem}.py::")
            )
    return related


def _changed_source_tests(change: dict, coverage_map: dict, test_ids: list[str]) -> set[str]:
    path = _normalized(change["old"] or change["new"])
    if path.startswith("tests/fixtures/"):
        return _fixture_tests(path, coverage_map.get("access", {}), test_ids)
    if path.startswith("tests/"):
        return _module_tests(path, test_ids)
    if not path.startswith("sqlglot/") or not path.endswith(".py"):
        return _fixture_tests(path, coverage_map.get("access", {}), test_ids)

    source = coverage_map.get("source", {})
    old_lines = change["old_lines"]
    new_lines = change["new_lines"]
    regions: list[tuple[int, int]] = []
    scopes = coverage_map.get("scopes", {}).get(path, [])

    for line in old_lines:
        containing = [(start, end) for start, end in scopes if start <= line <= end]
        if containing:
            regions.append(min(containing, key=lambda span: span[1] - span[0]))
        else:
            regions.append((line, line))

    if not old_lines and new_lines:
        # Added lines have no baseline coverage; use the nearest old source line.
        anchor = max(1, min(new_lines) - 1)
        containing = [(start, end) for start, end in scopes if start <= anchor <= end]
        regions.extend(containing or [(max(1, anchor - 1), anchor + 1)])

    selected = _tests_covering_region(source, path, regions)
    if not selected:
        selected = _module_tests(path, test_ids)
    if not selected:
        selected = set(source.get(path, {}))
    selected.update(_related_tests(path, test_ids))
    if not selected and path.startswith("sqlglot/dialects/"):
        selected = {
            test_id for test_id in test_ids
            if test_id.startswith("tests/dialects/")
        }
    return selected


def _test_number_map(tests: list[dict]) -> dict[str, int]:
    return {test["id"]: test["number"] for test in tests}


def _select_tests(patch: Path, coverage_map: dict, tests: list[dict]) -> list[dict]:
    ids = [test["id"] for test in tests]
    selected: set[str] = set()
    for change in _patch_changes(patch):
        selected.update(_changed_source_tests(change, coverage_map, ids))
    number_by_id = _test_number_map(tests)
    return [test for test in tests if test["id"] in selected and test["id"] in number_by_id]


def _split(tests: list[dict]) -> list[list[int]]:
    if not tests:
        return []

    durations = [int(test["seconds"] * 1000 + 0.5) for test in tests]
    best_workers: list[list[int]] = []
    best_cost: int | None = None
    for worker_count in range(1, min(8, len(tests)) + 1):
        workers: list[list[int]] = [[] for _ in range(worker_count)]
        loads = [0] * worker_count
        ordered = sorted(
            zip(tests, durations),
            key=lambda item: (-item[1], item[0]["number"]),
        )
        for test, duration in ordered:
            worker = min(range(worker_count), key=loads.__getitem__)
            workers[worker].append(test["number"])
            loads[worker] += duration
        cost = max(loads) + 2_000 * worker_count
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_workers = workers
    return best_workers


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    coverage_map = json.loads((APP / "coverage-map.json").read_text())
    for patch in sorted((DATA / "commits").glob("*.patch")):
        selected = _select_tests(patch, coverage_map, tests)
        print(json.dumps({"commit": patch.stem, "workers": _split(selected)}))
    print("selected tests using pinned-source coverage", file=sys.stderr)


if __name__ == "__main__":
    main()
