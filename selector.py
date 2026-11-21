"""Select tests by mapping changed source regions to tests that execute them."""

from __future__ import annotations

import json
import re
import sys
from bisect import bisect_left
from pathlib import Path, PurePosixPath
from source_analysis import SourceIndex
from coverage_index import LineIndex

DATA = Path("/data")
APP = Path("/app")
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _patch_changes(patch: Path) -> list[dict]:
    changes: list[dict] = []
    current: dict | None = None
    old_line = 0
    in_hunk = False
    replacing = False
    for line in patch.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            old_path = parts[2][2:] if len(parts) > 2 and parts[2].startswith("a/") else ""
            new_path = parts[3][2:] if len(parts) > 3 and parts[3].startswith("b/") else ""
            current = {
                "old": "" if old_path == "/dev/null" else old_path,
                "new": "" if new_path == "/dev/null" else new_path,
                "old_lines": set(),
                "insertion_anchors": set(),
                "added": [],
                "removed": [],
                "hunks": [],
            }
            changes.append(current)
            in_hunk = False
            continue
        if current is None:
            continue
        match = HUNK.match(line)
        if match:
            current["hunks"].append((int(match[1]), int(match[2] or 1), int(match[4] or 1), []))
            # A zero-length old range names the line BEFORE the insertion.
            old_line = int(match.group(1)) + (match.group(2) == "0")
            in_hunk = True
            replacing = False
            continue
        if not in_hunk or line.startswith("\\"):
            continue
        if line[:1] in {" ", "+", "-"}:
            current["hunks"][-1][3].append(line)
        if line.startswith("+"):
            if not replacing:
                current["insertion_anchors"].add(old_line)
            current["added"].append(line[1:])
        elif line.startswith("-"):
            replacing = True
            current["old_lines"].add(old_line)
            current["removed"].append(line[1:])
            old_line += 1
        elif line.startswith(" "):
            old_line += 1
            replacing = False
    return changes


def _normalized(path: str) -> str:
    return str(PurePosixPath(path))


def _tests_covering_region(source: dict, path: str, regions: list[tuple[int, int]]) -> set[str]:
    index = source.get(path)
    if isinstance(index, LineIndex):
        return index.covering(regions)
    merged: list[list[int]] = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    result: set[str] = set()
    for test_id, lines in source.get(path, {}).items():
        for start, end in merged:
            index = bisect_left(lines, start)
            if index < len(lines) and lines[index] <= end:
                result.add(test_id)
                break
    return result


def _function_tests(path: str, changed_lines: set[int], coverage_map: dict) -> set[str]:
    functions = coverage_map.get("functions", {})
    matching = [
        function
        for function in functions.get(path, [])
        if any(function["start"] <= line <= function["end"] for line in changed_lines)
    ]
    if not matching:
        return set()

    source = coverage_map.get("source", {})
    return _tests_covering_region(
        source,
        path,
        [(function["start"], function["end"]) for function in matching],
    )


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


def _changed_source_tests(change: dict, coverage_map: dict, test_ids: list[str], index=None) -> set[str]:
    path = _normalized(change["old"] or change["new"])
    if path.startswith("tests/fixtures/"):
        return _fixture_tests(path, coverage_map.get("access", {}), test_ids)
    if path.startswith("tests/"):
        return _module_tests(path, test_ids)
    if not path.startswith("sqlglot/") or not path.endswith(".py"):
        return _fixture_tests(path, coverage_map.get("access", {}), test_ids)

    changed_lines = change["old_lines"]
    lines = set(changed_lines)
    if index is not None:
        starts = index.conditions.get(path, {})
        lines.update(starts.get(str(line), line) for line in changed_lines)
    selected = _tests_covering_region(
        coverage_map.get("source", {}), path, [(line, line) for line in lines]
    )
    functions = coverage_map.get("functions", {}).get(path, ())
    in_body = changed_lines and all(
        any(f["start"] < line <= f["end"] for f in functions) for line in changed_lines
    )
    if in_body and not change["insertion_anchors"] and change["added"] and change["removed"]:
        return selected
    if not selected:
        selected.update(_function_tests(path, changed_lines, coverage_map))
    for anchor in change["insertion_anchors"]:
        # No baseline test could execute the added lines. Use the enclosing
        # function on either side of the gap, including at a function boundary.
        neighbors = {max(1, anchor - 1), anchor}
        affected = _function_tests(path, neighbors, coverage_map)
        if not affected:
            # Module-scope insertions can affect any consumer of this file.
            affected = set(coverage_map.get("source", {}).get(path, {}))
        selected.update(affected)
    dialect_key = re.sub(r"[^a-z0-9]", "", Path(path).stem.lower())
    selected.update(coverage_map.get("dialect_tests", {}).get(dialect_key, []))
    selected.update(_module_tests(path, test_ids))
    return selected


def _select_tests(
    patch: Path, coverage_map: dict, tests: list[dict], ids: list[str], index=None,
) -> list[dict]:
    selected: set[str] = set()
    changes = _patch_changes(patch)
    for change in changes:
        path = _normalized(change["old"] or change["new"])
        selected.update(coverage_map.get("access", {}).get(path, ()))
        if index is None or not index.equivalent(change):
            selected.update(_changed_source_tests(change, coverage_map, ids, index))
    return [test for test in tests if test["id"] in selected]


def _split(tests: list[dict]) -> list[list[int]]:
    if not tests:
        return []

    ordered = sorted(
        ((test["number"], int(test["seconds"] * 1000 + 0.5)) for test in tests),
        key=lambda item: (-item[1], item[0]),
    )
    total_duration = sum(duration for _, duration in ordered)
    longest = ordered[0][1]
    best_workers: list[list[int]] = []
    best_cost: int | None = None
    for worker_count in range(1, min(8, len(tests)) + 1):
        worker_cost = 2_000 * worker_count
        if best_cost is not None:
            if longest + worker_cost >= best_cost:
                break
            average = (total_duration + worker_count - 1) // worker_count
            if average + worker_cost >= best_cost:
                continue

        workers: list[list[int]] = [[] for _ in range(worker_count)]
        loads = [0] * worker_count
        for number, duration in ordered:
            worker = loads.index(min(loads))
            workers[worker].append(number)
            loads[worker] += duration
        cost = max(loads) + worker_cost
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_workers = workers
    return best_workers


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    ids = [test["id"] for test in tests]
    coverage_map = json.loads((APP / "coverage-map.json").read_text())
    if "line_index" in coverage_map:
        coverage_map["source"] = {p: LineIndex(v) for p, v in coverage_map.pop("line_index").items()}
    index = SourceIndex(APP / "source-index.zip")
    for patch in sorted((DATA / "commits").glob("*.patch")):
        selected = _select_tests(patch, coverage_map, tests, ids, index)
        print(json.dumps({"commit": patch.stem, "workers": _split(selected)}))
    print("selected tests using pinned-source coverage", file=sys.stderr)


if __name__ == "__main__":
    main()
