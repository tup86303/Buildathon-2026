"""Select tests by mapping changed source regions to tests that execute them."""

from __future__ import annotations

import ast
import json
import re
import sys
from bisect import bisect_left
from pathlib import Path, PurePosixPath
from source_analysis import SourceIndex
from coverage_index import LineIndex
from verifier import verify_and_prune

DATA = Path("/data")
APP = Path("/app")
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DIFF_GIT = re.compile(r'^diff --git ("(?:\\.|[^"\\])*"|a/.+?) ("(?:\\.|[^"\\])*"|b/.+)$')
BRANCH_HEADER_RE = re.compile(r"^\s*(?:elif\b|else\s*:|except\b|finally\s*:|case\b)")


def _unquote_git_path(raw: str, prefixed: bool = True) -> str:
    raw = raw.partition("\t")[0]
    if raw.startswith('"') and raw.endswith('"'):
        try:
            raw = ast.literal_eval("b" + raw).decode("utf-8", "surrogateescape")
        except (SyntaxError, ValueError):
            raw = raw[1:-1]
    if raw == "/dev/null":
        return ""
    if prefixed and raw.startswith(("a/", "b/")):
        return raw[2:]
    return raw


def _patch_changes(patch: Path) -> list[dict]:
    changes: list[dict] = []
    current: dict | None = None
    old_line = 0
    in_hunk = False
    replacing = False
    for line in patch.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("diff --git "):
            m = DIFF_GIT.match(line)
            if m:
                old_path = _unquote_git_path(m.group(1))
                new_path = _unquote_git_path(m.group(2))
            else:
                parts = line.split()
                old_path = _unquote_git_path(parts[2]) if len(parts) > 2 else ""
                new_path = _unquote_git_path(parts[3]) if len(parts) > 3 else ""
            current = {
                "old": old_path,
                "new": new_path,
                "is_deleted": False,
                "is_new": False,
                "is_rename": False,
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
        if not in_hunk:
            if line.startswith("deleted file mode "):
                current["is_deleted"] = True
                continue
            if line.startswith("new file mode "):
                current["is_new"] = True
                continue
            if line.startswith("rename from "):
                current["is_rename"] = True
                current["old"] = _unquote_git_path(line[len("rename from "):], prefixed=False)
                continue
            if line.startswith("rename to "):
                current["is_rename"] = True
                current["new"] = _unquote_git_path(line[len("rename to "):], prefixed=False)
                continue
            if line.startswith("--- "):
                old_p = _unquote_git_path(line[4:])
                if old_p == "":
                    current["old"] = ""
                    current["is_new"] = True
                else:
                    current["old"] = old_p
                continue
            if line.startswith("+++ "):
                new_p = _unquote_git_path(line[4:])
                if new_p == "":
                    current["new"] = ""
                    current["is_deleted"] = True
                else:
                    current["new"] = new_p
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
    return str(PurePosixPath(path)) if path else ""


def _get_source_entry(source: dict, path: str):
    entry = source.get(path)
    if isinstance(entry, list):
        entry = LineIndex(entry)
        source[path] = entry
    return entry


def _tests_covering_region(source: dict, path: str, regions: list[tuple[int, int]]) -> set[str]:
    index = _get_source_entry(source, path)
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
            idx = bisect_left(lines, start)
            if idx < len(lines) and lines[idx] <= end:
                result.add(test_id)
                break
    return result


def _file_tests(path: str, coverage_map: dict) -> set[str]:
    entry = _get_source_entry(coverage_map.get("source", {}), path)
    if entry is None:
        return set()
    return set(entry)


def _dependent_tests(path: str, coverage_map: dict, test_ids: list[str], index=None) -> set[str]:
    if index is None:
        return set()
    deps = index.module_dependents.get(path)
    if not deps:
        return set()
    result: set[str] = set()
    for dep_src in deps.get("sqlglot", ()):
        result.update(_file_tests(dep_src, coverage_map))
        dialect_key = re.sub(r"[^a-z0-9]", "", Path(dep_src).stem.lower())
        result.update(coverage_map.get("dialect_tests", {}).get(dialect_key, []))
        result.update(_module_tests(dep_src, test_ids))
    dep_test_files = set(deps.get("tests", ()))
    if dep_test_files:
        for test_id in test_ids:
            test_file = test_id.partition("::")[0]
            if test_file in dep_test_files:
                result.add(test_id)
    return result


def _function_tests(path: str, changed_lines: set[int], coverage_map: dict, innermost_only: bool = False) -> set[str]:
    functions = coverage_map.get("functions", {}).get(path, [])
    matching = [
        function
        for function in functions
        if any(function["start"] <= line <= function["end"] for line in changed_lines)
    ]
    if not matching:
        return set()

    if innermost_only:
        chosen = []
        for line in changed_lines:
            enclosing = [f for f in matching if f["start"] <= line <= f["end"]]
            if enclosing:
                chosen.append(min(enclosing, key=lambda f: f["end"] - f["start"]))
        if chosen:
            result = _tests_covering_region(
                coverage_map.get("source", {}),
                path,
                [(f["start"], f["end"]) for f in chosen],
            )
            if result:
                return result

    return _tests_covering_region(
        coverage_map.get("source", {}),
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
    if not result:
        return set(test_ids)
    return result


def _expr_symbol_tests(path: str, changed_lines: set[int], coverage_map: dict, index) -> set[str] | None:
    if index is None or not path.startswith("sqlglot/expressions/") or path.endswith("__init__.py") or not changed_lines:
        return None
    sym_data = getattr(index, "expr_symbols", None)
    if not sym_data:
        return None
    defs = sym_data.get("classes", {}).get(path, ())
    if not defs:
        return None
    matched = set()
    for ln in changed_lines:
        hits = [name for s, e, name in defs if s <= ln <= e]
        if not hits:
            return None
        matched.update(hits)
    func_classes = set(sym_data.get("func_classes", ()))
    if matched & func_classes:
        return set()  # sentinel: metaclass-registered Func subclass -> caller must widen
    refs_map = sym_data.get("refs", {})
    if not matched or any(name not in refs_map for name in matched):
        return None
    source = coverage_map.get("source", {})
    selected: set[str] = set()
    for name in matched:
        for ref_file, regions in refs_map[name].items():
            selected.update(_tests_covering_region(source, ref_file, [(s, e) for s, e in regions]))
    return selected if selected else None


def _changed_source_tests(
    change: dict, coverage_map: dict, test_ids: list[str], index=None, analysis: str = "unknown"
) -> set[str]:
    path = _normalized(change["old"] or change["new"])
    if not path:
        return set(test_ids)
    if path.startswith("tests/fixtures/"):
        return _fixture_tests(path, coverage_map.get("access", {}), test_ids)
    if path.startswith("tests/"):
        dep_tests = _dependent_tests(path, coverage_map, test_ids, index)
        test_status = "unknown"
        if index is not None and not change.get("is_deleted") and not change.get("is_new") and not change.get("is_rename"):
            sq_root = APP / "sqlglot_src"
            if sq_root.is_dir():
                test_status = index.check_test_file(change, sq_root)
                if test_status == "equivalent":
                    return set()
        if (
            index is not None
            and test_status != "syntax_error"
            and not dep_tests
            and Path(path).name.startswith("test_")
            and not change.get("is_deleted")
            and not change.get("is_new")
            and not change.get("is_rename")
        ):
            methods = index.test_methods.get(path, ())
            old_lines = change["old_lines"]
            anchors = change["insertion_anchors"]
            if methods and (old_lines or anchors):
                # Only narrow if EVERY changed line and insertion anchor falls strictly inside
                # a known test_* method body (never def headers, setUp, helpers, or module scope).
                id_set = set(test_ids)
                matched: set[str] = set()
                all_inside = True
                for ln in old_lines:
                    hits = [tid for s, e, tid in methods if s <= ln <= e and tid in id_set]
                    if not hits:
                        all_inside = False
                        break
                    matched.update(hits)
                if all_inside:
                    for anc in anchors:
                        hits = [tid for s, e, tid in methods if s <= anc <= e and tid in id_set]
                        if not hits:
                            all_inside = False
                            break
                        matched.update(hits)
                if all_inside and matched:
                    return matched
        selected = _module_tests(path, test_ids)
        selected.update(dep_tests)
        if Path(path).name == "conftest.py" or (not selected and path.endswith(".py")):
            prefix = str(PurePosixPath(path).parent)
            prefix = "" if prefix == "." else prefix + "/"
            selected.update(t for t in test_ids if t.startswith(prefix))
        return selected
    if not path.startswith("sqlglot/") or not path.endswith(".py"):
        result = _fixture_tests(path, coverage_map.get("access", {}), test_ids)
        if Path(path).name in {"pyproject.toml", "setup.py", "setup.cfg", "pytest.ini"} or (path.endswith(".py") and not result):
            return set(test_ids)
        return result

    dialect_key = re.sub(r"[^a-z0-9]", "", Path(path).stem.lower())
    base_extra = set(coverage_map.get("dialect_tests", {}).get(dialect_key, []))
    base_extra.update(_module_tests(path, test_ids))

    # Syntax errors or package __init__.py re-export edits in sqlglot/ can break imports across the suite.
    if analysis == "syntax_error" or Path(path).name == "__init__.py":
        return set(test_ids)
    if (
        analysis == "unknown"
        or change.get("is_deleted")
        or change.get("is_rename")
        or change.get("is_new")
    ):
        selected = _file_tests(path, coverage_map)
        selected.update(_dependent_tests(path, coverage_map, test_ids, index))
        selected.update(base_extra)
        return selected or set(test_ids)

    if index is not None and analysis in {"local_expr", "structural"} and len(change.get("hunks", ())) >= 2:
        changed_lines, filtered_anchors = index.filter_unchanged_function_lines(change)
        if not changed_lines and not filtered_anchors:
            return set()
        change = {**change, "old_lines": changed_lines, "insertion_anchors": filtered_anchors}
    else:
        changed_lines = change["old_lines"]
    lines = set(changed_lines)
    if index is not None:
        starts = index.conditions.get(path, {})
        for line in changed_lines:
            extra = starts.get(str(line))
            if isinstance(extra, list):
                lines.update(extra)
            elif isinstance(extra, int):
                lines.add(extra)
    selected = _tests_covering_region(
        coverage_map.get("source", {}), path, [(line, line) for line in lines]
    )
    if index is not None and changed_lines:
        dp = getattr(index, "dialect_profile", None) or {}
        for d_rel, file_lines in dp.get("lazy_import_lines", {}).items():
            traced = file_lines.get(path)
            if traced and (changed_lines & set(traced)):
                selected.update(dp.get("dialect_tests", {}).get(d_rel, ()))
    functions = coverage_map.get("functions", {}).get(path, ())
    header_ranges = index.def_headers.get(path, ()) if index is not None else ()
    in_def_header = any(
        any(s <= line <= e for s, e in header_ranges) for line in changed_lines
    )
    in_body = bool(changed_lines) and not in_def_header and all(
        any(f["start"] < line <= f["end"] for f in functions) for line in changed_lines
    )

    if in_body and not change["insertion_anchors"] and change["added"] and change["removed"]:
        edits = change["added"] + change["removed"]
        branch_header_edit = any(BRANCH_HEADER_RE.match(s) for s in edits)
        if not selected or analysis == "structural" or branch_header_edit:
            selected.update(_function_tests(path, changed_lines, coverage_map, innermost_only=True))
            if not selected:
                selected.update(_file_tests(path, coverage_map))
                selected.update(_dependent_tests(path, coverage_map, test_ids, index))
                selected.update(base_extra)
        return selected or set(test_ids)

    if in_def_header and changed_lines:
        selected.update(_function_tests(path, changed_lines, coverage_map, innermost_only=True))
    if (not selected or analysis == "structural") and changed_lines:
        selected.update(_function_tests(path, changed_lines, coverage_map, innermost_only=bool(selected)))

    # Module-scope or class-scope replacement (e.g. TRANSFORMS, KEYWORDS, FUNCTIONS, arg_types):
    # these execute at import time and have 0 runtime line coverage.
    if changed_lines and not in_body:
        has_module_or_class_line = any(
            not any(f["start"] < line <= f["end"] for f in functions) for line in changed_lines
        )
        if has_module_or_class_line:
            expr_hits = _expr_symbol_tests(path, changed_lines, coverage_map, index)
            if expr_hits is not None and not expr_hits:
                # Metaclass-registered Func subclass (Parser.FUNCTIONS dynamic dispatch): widen safely!
                return set(test_ids)
            if expr_hits is not None and not change["insertion_anchors"]:
                selected = set(expr_hits)
                selected.update(base_extra)
                return selected
            if (
                index is not None
                and path.startswith("sqlglot/dialects/")
                and not path.endswith(("__init__.py", "/dialect.py"))
                and not change["insertion_anchors"]
            ):
                bindings = index.scalar_bindings.get(path, {})
                bound_names = [bindings.get(str(line)) for line in changed_lines]
                if bound_names and all(n is not None for n in bound_names):
                    # Replacing a literal assignment with executable code can
                    # cause import errors before any tracked attribute read.
                    replacement_names = []
                    for line in change["added"]:
                        try:
                            statements = ast.parse(line.strip()).body
                        except SyntaxError:
                            return set(test_ids)
                        if len(statements) != 1:
                            return set(test_ids)
                        assignment = statements[0]
                        if (not isinstance(assignment, ast.Assign)
                                or len(assignment.targets) != 1
                                or not isinstance(assignment.targets[0], ast.Name)):
                            return set(test_ids)
                        try:
                            literal = ast.literal_eval(assignment.value)
                        except (SyntaxError, ValueError):
                            return set(test_ids)
                        if not isinstance(literal, (str, bytes, bool, int, float, complex, type(None))):
                            return set(test_ids)
                        replacement_names.append(assignment.targets[0].id)
                    if sorted(replacement_names) != sorted(bound_names):
                        return set(test_ids)
                    attr_hits: set[str] = set()
                    all_have_readers = True
                    for name in set(bound_names):
                        readers = index.read_sites.get(name, {})
                        if not readers:
                            all_have_readers = False
                            break
                        for reader, sites in readers.items():
                            if reader.startswith("tests/"):
                                attr_hits.update(_module_tests(reader, test_ids))
                            else:
                                attr_hits.update(_tests_covering_region(coverage_map["source"], reader, [(n, n) for n in sites]))
                    if all_have_readers and attr_hits:
                        dp = getattr(index, "dialect_profile", None) or {}
                        d_users = dp.get("dialect_tests", {}).get(path)
                        if d_users is not None:
                            allowed = set(d_users) | set(dp.get("unprofiled", ()))
                            narrowed = attr_hits & allowed
                            if narrowed:
                                return narrowed
                        selected = set(attr_hits)
                        selected.update(_module_tests(path, test_ids))
                        return selected
            selected.update(_file_tests(path, coverage_map))
            selected.update(_dependent_tests(path, coverage_map, test_ids, index))
            # Non-scalar class/module edits in dialect/parser/generator files (e.g. KEYWORDS, FUNCTIONS,
            # TRANSFORMS, TYPE_MAPPING) can affect cross-dialect tests that never execute runtime lines
            # in the edited file; include all MRO-matched runtime dialect users from dialect-profile.
            if index is not None and path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
                dp = getattr(index, "dialect_profile", None) or {}
                d_rel = f"sqlglot/dialects/{Path(path).name}"
                d_users = dp.get("dialect_tests", {}).get(d_rel)
                if d_users:
                    # test_generator_memory only instantiates dialect.generator() in a loop (no tokenizer/parser)
                    is_gen_file = path.startswith("sqlglot/generators/")
                    if not is_gen_file:
                        selected.update(t for t in d_users if not t.endswith("::test_generator_memory"))
                    else:
                        selected.update(d_users)
                    selected.update(dp.get("unprofiled", ()))
                if path.startswith("sqlglot/typing/"):
                    selected.update(t for t in test_ids if t.endswith(("::test_annotate_types", "::test_annotate_funcs")))
            if index is not None:
                bindings = index.scalar_bindings.get(path, {})
                for name in {bindings.get(str(line)) for line in changed_lines} - {None}:
                    for reader, sites in index.read_sites.get(name, {}).items():
                        if reader.startswith("tests/"):
                            selected.update(_module_tests(reader, test_ids))
                        else:
                            selected.update(_tests_covering_region(coverage_map["source"], reader, [(n, n) for n in sites]))

    for anchor in change["insertion_anchors"]:
        # No baseline test could execute the added lines. Use the enclosing
        # function on either side of the gap, including at a function boundary.
        neighbors = {max(1, anchor - 1), anchor}
        affected = _function_tests(path, neighbors, coverage_map)
        if not affected:
            # Module/class-scope insertions can affect any consumer of this file (and cross-dialect users).
            affected = _file_tests(path, coverage_map)
            affected.update(_dependent_tests(path, coverage_map, test_ids, index))
            if index is not None and path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
                dp = getattr(index, "dialect_profile", None) or {}
                d_rel = f"sqlglot/dialects/{Path(path).name}"
                d_users = dp.get("dialect_tests", {}).get(d_rel)
                if d_users:
                    if not path.startswith("sqlglot/generators/"):
                        affected.update(t for t in d_users if not t.endswith("::test_generator_memory"))
                    else:
                        affected.update(d_users)
                    affected.update(dp.get("unprofiled", ()))
                if path.startswith("sqlglot/typing/"):
                    affected.update(t for t in test_ids if t.endswith(("::test_annotate_types", "::test_annotate_funcs")))
        selected.update(affected)

    selected.update(base_extra)
    if not selected:
        selected.update(_file_tests(path, coverage_map))
        selected.update(_dependent_tests(path, coverage_map, test_ids, index))
    return selected or set(test_ids)


def _select_tests(
    patch: Path, coverage_map: dict, tests: list[dict], ids: list[str], index=None,
) -> list[dict]:
    selected: set[str] = set()
    changes = _patch_changes(patch)
    if not changes:
        return tests
    for change in changes:
        old_path = _normalized(change["old"])
        new_path = _normalized(change["new"])
        for p in {old_path, new_path} - {""}:
            selected.update(coverage_map.get("access", {}).get(p, ()))
        if old_path.startswith("sqlglot/") and index and any(
            marker in s for s in change["added"] + change["removed"] for marker in ('"""', "'''", ">>>")
        ):
            for suite in index.doctest_suites:
                selected.update(_module_tests(suite, ids))
        analysis = index.analyze(change) if index is not None else "unknown"
        if analysis != "equivalent":
            selected.update(_changed_source_tests(change, coverage_map, ids, index, analysis))
            if new_path and new_path != old_path and old_path:
                selected.update(_changed_source_tests({**change, "old": new_path}, coverage_map, ids, index))
            if index is not None:
                for path in {old_path, new_path}:
                    selected.update(index.subprocess_readers.get(path, ()))
    return [test for test in tests if test["id"] in selected]


def _refine_schedule(workers: list[list[int]], loads: list[int], dur_map: dict[int, int], lower_bound: int) -> None:
    """Bounded local search: move or swap a test off the busiest worker if it strictly reduces makespan."""
    k = len(workers)
    if k <= 1:
        return
    for _ in range(30):
        max_w = max(range(k), key=loads.__getitem__)
        max_load = loads[max_w]
        if max_load <= lower_bound:
            break
        improved = False
        min_w = min(range(k), key=loads.__getitem__)
        for idx, t in enumerate(workers[max_w]):
            d = dur_map[t]
            if loads[min_w] + d < max_load:
                workers[max_w].pop(idx)
                workers[min_w].append(t)
                loads[max_w] -= d
                loads[min_w] += d
                improved = True
                break
        if improved:
            continue
        for other_w in range(k):
            if other_w == max_w:
                continue
            for i, t1 in enumerate(workers[max_w]):
                d1 = dur_map[t1]
                for j, t2 in enumerate(workers[other_w]):
                    d2 = dur_map[t2]
                    if d1 > d2 and max(loads[max_w] - d1 + d2, loads[other_w] - d2 + d1) < max_load:
                        workers[max_w][i], workers[other_w][j] = t2, t1
                        loads[max_w] += d2 - d1
                        loads[other_w] += d1 - d2
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break


def _split(tests: list[dict]) -> list[list[int]]:
    if not tests:
        return []

    ordered = sorted(
        ((test["number"], int(test["seconds"] * 1000 + 0.5)) for test in tests),
        key=lambda item: (-item[1], item[0]),
    )
    dur_map = dict(ordered)
    total_duration = sum(duration for _, duration in ordered)
    longest = ordered[0][1]
    best_workers: list[list[int]] = []
    best_cost: int | None = None
    for worker_count in range(1, min(8, len(tests)) + 1):
        worker_cost = 2_000 * worker_count
        average = (total_duration + worker_count - 1) // worker_count
        if best_cost is not None:
            if longest + worker_cost >= best_cost:
                break
            if average + worker_cost >= best_cost:
                continue

        workers: list[list[int]] = [[] for _ in range(worker_count)]
        loads = [0] * worker_count
        for number, duration in ordered:
            worker = loads.index(min(loads))
            workers[worker].append(number)
            loads[worker] += duration
        if max(loads) > max(longest, average):
            _refine_schedule(workers, loads, dur_map, max(longest, average))
        cost = max(loads) + worker_cost
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_workers = workers
    return best_workers


def _schedule_cost_ms(workers: list[list[int]], dur_map: dict[int, int]) -> int:
    if not workers:
        return 0
    return max(sum(dur_map[n] for n in w) for w in workers) + 2_000 * len(workers)


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    ids = [test["id"] for test in tests]
    dur_map = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in tests}
    coverage_map = json.loads((APP / "coverage-map.json").read_text())
    if "line_index" in coverage_map:
        coverage_map["source"] = coverage_map.pop("line_index")
    index = SourceIndex(APP / "source-index.zip")
    sqlglot_root = APP / "sqlglot_src"
    for patch in sorted((DATA / "commits").glob("*.patch")):
        changes = _patch_changes(patch)
        selected = _select_tests(patch, coverage_map, tests, ids, index)
        workers = _split(selected)
        if selected and sqlglot_root.exists():
            est_ms = _schedule_cost_ms(workers, dur_map)
            pruned = verify_and_prune(sqlglot_root, changes, index, selected, est_ms)
            if len(pruned) != len(selected):
                workers = _split(pruned)
        print(json.dumps({"commit": patch.stem, "workers": workers}))
    print("selected tests using pinned-source coverage", file=sys.stderr)


if __name__ == "__main__":
    main()
