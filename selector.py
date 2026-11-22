"""Select tests to run for each commit patch using pinned-source coverage and static/runtime profiles."""

from __future__ import annotations

import ast
import gzip
import json
import marshal
import os
import re
import sys
from pathlib import Path, PurePosixPath

from coverage_index import LineIndex
from fast_verifier import probe_import_crash, verify_commit_unified
from predicate_eval import prune_unaffected_tests
from source_analysis import SourceIndex

APP = Path("/app")
DATA = Path("/data")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
BRANCH_HEADER_RE = re.compile(r"^\s*(?:elif\b|else\s*:|except\b|finally\s*:|case\b)")


def _normalized(path: str | None) -> str:
    if not path or path == "/dev/null":
        return ""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _patch_changes(patch: Path) -> list[dict]:
    changes: list[dict] = []
    current: dict | None = None
    in_hunk = False
    old_line = 0
    for line in patch.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("diff --git "):
            current = {
                "old": "", "new": "", "old_lines": set(), "insertion_anchors": [],
                "added": [], "removed": [], "hunks": [],
                "is_deleted": False, "is_new": False, "is_rename": False,
            }
            changes.append(current)
            in_hunk = False
            continue
        if current is None:
            continue
        if line.startswith("deleted file mode"):
            current["is_deleted"] = True
        elif line.startswith("new file mode"):
            current["is_new"] = True
        elif line.startswith(("rename from ", "rename to ")):
            current["is_rename"] = True
        elif line.startswith("--- "):
            current["old"] = _normalized(line[4:].strip())
        elif line.startswith("+++ "):
            current["new"] = _normalized(line[4:].strip())
        elif line.startswith("@@ "):
            match = HUNK_RE.match(line)
            if match:
                old_line = int(match.group(1))
                old_count = 1 if match.group(2) is None else int(match.group(2))
                new_count = 1 if match.group(4) is None else int(match.group(4))
                current["hunks"].append((old_line, old_count, new_count, []))
                in_hunk = True
        elif in_hunk:
            if line.startswith("-"):
                current["old_lines"].add(old_line)
                current["removed"].append(line[1:])
                current["hunks"][-1][3].append(line)
                current["_had_del"] = True
                old_line += 1
            elif line.startswith("+"):
                if not current.get("_had_del"):
                    current["insertion_anchors"].append(old_line)
                current["added"].append(line[1:])
                current["hunks"][-1][3].append(line)
            elif line.startswith(" "):
                current["_had_del"] = False
                current["hunks"][-1][3].append(line)
                old_line += 1
            elif line.startswith("\\"):
                continue
            else:
                in_hunk = False
    return changes


def _tests_covering_region(source: dict, path: str, regions: list[tuple[int, int]]) -> set[str]:
    file_tests = source.get(path)
    if not file_tests or not regions:
        return set()
    if isinstance(file_tests, LineIndex):
        return file_tests.covering(regions)
    selected: set[str] = set()
    for test_id, covered in file_tests.items():
        if any(start <= line <= end for line in covered for start, end in regions):
            selected.add(test_id)
    return selected


def _function_tests(path: str, lines: set[int], coverage_map: dict, innermost_only: bool = False) -> set[str]:
    functions = coverage_map.get("functions", {}).get(path, ())
    regions = []
    for line in lines:
        enclosing = [f for f in functions if f["start"] <= line <= f["end"]]
        if not enclosing:
            continue
        if innermost_only:
            f = min(enclosing, key=lambda x: x["end"] - x["start"])
            regions.append((f["start"], f["end"]))
        else:
            for f in enclosing:
                regions.append((f["start"], f["end"]))
    return _tests_covering_region(coverage_map.get("source", {}), path, regions)


def _file_tests(path: str, coverage_map: dict) -> set[str]:
    file_cov = coverage_map.get("source", {}).get(path)
    if not file_cov:
        return set()
    return set(file_cov)


def _module_tests(path: str, test_ids: list[str]) -> set[str]:
    if path.startswith("tests/") and path.endswith(".py"):
        prefix = f"{path}::"
        return {test_id for test_id in test_ids if test_id.startswith(prefix)}
    stem = Path(path).stem
    candidates = {f"tests/test_{stem}.py::", f"tests/dialects/test_{stem}.py::"}
    return {test_id for test_id in test_ids if any(test_id.startswith(prefix) for prefix in candidates)}


def _fixture_tests(path: str, access: dict, test_ids: list[str]) -> set[str]:
    result = set(access.get(path, ()))
    if result:
        return result
    parts = PurePosixPath(path).parts
    if len(parts) > 3:
        category = parts[2]
        result.update(
            test_id for test_id in test_ids
            if test_id.startswith(f"tests/{category}/")
            or test_id.startswith(f"tests/test_{category}.py::")
        )
    return result or set(test_ids)


def _scalar_assignments(lines: list[str]) -> list[str] | None:
    """Recognize plain scalar assignments whose RHS cannot fail during import."""
    names = []
    for line in lines:
        try:
            statements = ast.parse(line.strip()).body
        except (SyntaxError, ValueError):
            return None
        if len(statements) != 1 or not isinstance(statements[0], ast.Assign):
            return None
        assignment = statements[0]
        if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
            return None
        value = assignment.value
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
            value = value.operand
            if not isinstance(value, ast.Constant) or type(value.value) not in (int, float):
                return None
        if not isinstance(value, ast.Constant) or type(value.value) not in (str, bool, int, float, type(None)):
            return None
        names.append(assignment.targets[0].id)
    return names


def _expr_symbol_tests(path: str, changed_lines: set[int], coverage_map: dict, index: SourceIndex | None) -> set[str] | None:
    if index is None or not path.startswith("sqlglot/expressions/") or path.endswith("__init__.py") or not changed_lines:
        return None
    sym_data = index.expr_symbols
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
    func_sql_names = sym_data.get("func_sql_names", {})
    refs_map = sym_data.get("refs", {})
    if not matched or any(name not in refs_map for name in matched):
        return None
    source = coverage_map.get("source", {})
    selected: set[str] = set()
    for name in matched:
        for ref_file, regions in refs_map[name].items():
            selected.update(_tests_covering_region(source, ref_file, [(s, e) for s, e in regions]))
        if name in func_classes:
            # Metaclass-registered Func subclass: union all Parser.FUNCTIONS + Generator.TRANSFORMS runtime users!
            sql_names = func_sql_names.get(name, [name.upper()])
            d_dict = coverage_map.get("dict_key_tests", {})
            for d_rel, amap in d_dict.items():
                fmap = amap.get("FUNCTIONS", {})
                if fmap:
                    selected.update(fmap.get("*", ()))
                    for sn in sql_names:
                        selected.update(fmap.get(f'const:"{sn}"', ()))
                tmap = amap.get("TRANSFORMS", {})
                if tmap:
                    selected.update(tmap.get("*", ()))
                    selected.update(tmap.get(f"type:{name}", ()))
            selected.update(coverage_map.get("dialect_unprofiled", ()))
            selected.update(coverage_map.get("attr_unprofiled", ()))
    return selected if selected else None


def _component_dialect_users(path: str, changed_lines: set[int], coverage_map: dict, index: SourceIndex | None) -> set[str]:
    d_rel = f"sqlglot/dialects/{Path(path).name}"
    if path.startswith("sqlglot/parsers/"):
        users = coverage_map.get("parser_users", {}).get(d_rel) or coverage_map.get("dialect_users", {}).get(d_rel, ())
    elif path.startswith("sqlglot/generators/"):
        users = coverage_map.get("generator_users", {}).get(d_rel) or coverage_map.get("dialect_users", {}).get(d_rel, ())
    elif path.startswith("sqlglot/dialects/") and index is not None and changed_lines:
        tr = index.tokenizer_ranges.get(path, ())
        if tr and all(any(s <= ln <= e for s, e in tr) for ln in changed_lines):
            users = coverage_map.get("tokenizer_users", {}).get(d_rel) or coverage_map.get("dialect_users", {}).get(d_rel, ())
        else:
            users = coverage_map.get("dialect_users", {}).get(d_rel, ())
    else:
        users = coverage_map.get("dialect_users", {}).get(d_rel, ())
    out = set(users)
    if out:
        out.update(coverage_map.get("dialect_unprofiled", ()))
    return out


def _changed_source_tests(
    change: dict, coverage_map: dict, test_ids: list[str], index: SourceIndex | None = None, analysis: str = "unknown"
) -> set[str]:
    path = _normalized(change["old"] or change["new"])
    if not path:
        return set(test_ids)
    if path.startswith("tests/fixtures/"):
        return _fixture_tests(path, coverage_map.get("access", {}), test_ids)
    if path.startswith("tests/"):
        importers = index.test_importers.get(path, ()) if index is not None else ()
        test_status = "unknown"
        if index is not None and not change.get("is_deleted") and not change.get("is_new") and not change.get("is_rename"):
            sq_root = APP / "sqlglot_src"
            if sq_root.is_dir():
                test_status = index.check_test_file(change, sq_root)
                if test_status == "equivalent":
                    return set()
        if False and (
            index is not None
            and test_status != "invalid"
            and not importers
            and Path(path).name.startswith("test_")
            and not change.get("is_deleted")
            and not change.get("is_new")
            and not change.get("is_rename")
        ):
            methods = index.test_methods.get(path, ())
            old_lines = change["old_lines"]
            anchors = change["insertion_anchors"]
            if methods and (old_lines or anchors):
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
        for imp in importers:
            selected.update(_module_tests(imp, test_ids))
        if not selected and path.endswith(".py"):
            prefix = str(PurePosixPath(path).parent)
            prefix = "" if prefix == "." else prefix + "/"
            selected.update(t for t in test_ids if t.startswith(prefix))
        pred_prof = coverage_map.get("predicate_profile")
        if pred_prof and index is not None and selected and index._last_parsed is not None and index._last_parsed[0] == path:
            try:
                before_t = ast.parse(index.archive.read(path), type_comments=True)
                selected = prune_unaffected_tests(path, before_t, index._last_parsed[2], selected, pred_prof)
            except BaseException:
                pass
        return selected or set(test_ids)

    if not path.startswith("sqlglot/") or not path.endswith(".py"):
        result = _fixture_tests(path, coverage_map.get("access", {}), test_ids)
        if Path(path).name in {"pyproject.toml", "setup.py", "setup.cfg", "pytest.ini"} or (path.endswith(".py") and not result):
            return set(test_ids)
        return result

    # Package __init__.py re-exports or deleted/renamed/new files widen safely.
    if Path(path).name == "__init__.py" or change.get("is_deleted") or change.get("is_rename") or change.get("is_new"):
        return set(test_ids)

    dialect_key = re.sub(r"[^a-z0-9]", "", Path(path).stem.lower())
    base_extra = set(coverage_map.get("dialect_tests", {}).get(dialect_key, []))
    base_extra.update(_module_tests(path, test_ids))

    if index is not None and analysis in {"changed", "structural"} and len(change.get("hunks", ())) >= 2:
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
            if isinstance(extra, int):
                lines.add(extra)
            elif isinstance(extra, list):
                lines.update(extra)

    source_cov = coverage_map.get("source", {})
    if index is not None:
        s_starts = index.stmt_starts.get(path, {})
        if s_starts:
            for line in list(changed_lines):
                if not _tests_covering_region(source_cov, path, [(line, line)]):
                    st_ln = s_starts.get(str(line))
                    if isinstance(st_ln, int):
                        lines.add(st_ln)
    selected = _tests_covering_region(source_cov, path, [(line, line) for line in lines])

    # Lazy-import crash-cascade protection: only widen to tests/dialects/ on structural/module-scope changes
    # (pure in-body expression edits like core.py:929 are already covered by line_index without dumping 869 unrelated dialect tests!)
    if changed_lines and analysis == "structural":
        for d_rel, file_lines in coverage_map.get("lazy_import_lines", {}).items():
            traced = file_lines.get(path)
            if traced and (changed_lines & set(traced)):
                selected.update(coverage_map.get("dialect_users", {}).get(d_rel, ()))
                selected.update(t for t in test_ids if t.startswith("tests/dialects/"))

    functions = coverage_map.get("functions", {}).get(path, ())
    header_ranges = index.def_headers.get(path, ()) if index is not None else ()
    in_def_header = any(any(s <= line <= e for s, e in header_ranges) for line in changed_lines)
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
                selected.update(base_extra)
        return selected or set(test_ids)

    if in_def_header and changed_lines:
        selected.update(_function_tests(path, changed_lines, coverage_map, innermost_only=True))
    if (not selected or analysis == "structural") and changed_lines:
        selected.update(_function_tests(path, changed_lines, coverage_map, innermost_only=bool(selected)))

    # Module-scope or class-scope edits
    if changed_lines and not in_body:
        has_module_or_class_line = any(
            not any(f["start"] < line <= f["end"] for f in functions) for line in changed_lines
        )
        if has_module_or_class_line:
            expr_hits = _expr_symbol_tests(path, changed_lines, coverage_map, index)
            if expr_hits is not None and not expr_hits:
                return set(test_ids)
            if expr_hits is not None and not change["insertion_anchors"]:
                selected = set(expr_hits)
                selected.update(base_extra)
                return selected

            # Exact value-only TrackedDict key lookup for dialect/parser/generator dicts
            if (
                index is not None
                and path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/"))
                and Path(path).name not in {"dialect.py", "parser.py", "generator.py", "__init__.py"}
            ):
                tracked_attrs = set(coverage_map.get("tracked_dict_attrs", ()))
                d_rel = f"sqlglot/dialects/{Path(path).name}"
                d_dict_prof = coverage_map.get("dict_key_tests", {}).get(d_rel, {})
                if tracked_attrs and d_dict_prof:
                    dvc = index.dict_value_changes(change, tracked_attrs)
                    if dvc is not None:
                        d_attr, changed_keys = dvc
                        if d_attr in d_dict_prof:
                            key_map = d_dict_prof[d_attr]
                            out_d: set[str] = set(key_map.get("*", ()))
                            for ck in changed_keys:
                                out_d.update(key_map.get(ck, ()))
                            out_d.update(coverage_map.get("dialect_unprofiled", ()))
                            out_d.update(coverage_map.get("attr_unprofiled", ()))
                            return out_d

            # Exact runtime per-(dialect, attr) lookup + component-scoped scalar flag fallback
            if (
                index is not None
                and path.startswith("sqlglot/dialects/")
                and not path.endswith("/dialect.py")
                and not change["insertion_anchors"]
            ):
                bindings = index.scalar_bindings.get(path, {})
                bound_names = [bindings.get(str(line)) for line in changed_lines]
                if bound_names and all(n is not None for n in bound_names):
                    removed_names = _scalar_assignments(change["removed"])
                    added_names = _scalar_assignments(change["added"])
                    if not removed_names or removed_names != added_names:
                        return set(test_ids)
                    tr = index.tokenizer_ranges.get(path, ())
                    in_tok = bool(tr) and any(any(s <= ln <= e for s, e in tr) for ln in changed_lines)
                    watched = set(coverage_map.get("watched_dialect_attrs", ()))
                    if not in_tok and watched and all(n in watched for n in bound_names):
                        attr_prof = coverage_map.get("dialect_attr_tests", {}).get(path, {})
                        out_a: set[str] = set()
                        for n in set(bound_names):
                            out_a.update(attr_prof.get(n, ()))
                        out_a.update(coverage_map.get("dialect_unprofiled", ()))
                        out_a.update(coverage_map.get("attr_unprofiled", ()))
                        return out_a

                    d_u = set(coverage_map.get("dialect_users", {}).get(path, ()))
                    unprof = set(coverage_map.get("dialect_unprofiled", ()))
                    if d_u:
                        d_u |= unprof
                        p_u = (set(coverage_map.get("parser_users", {}).get(path, ())) | unprof) if coverage_map.get("parser_users", {}).get(path) else d_u
                        g_u = (set(coverage_map.get("generator_users", {}).get(path, ())) | unprof) if coverage_map.get("generator_users", {}).get(path) else d_u
                        t_u = (set(coverage_map.get("tokenizer_users", {}).get(path, ())) | unprof) if coverage_map.get("tokenizer_users", {}).get(path) else d_u
                    else:
                        p_u = g_u = t_u = None
                    attr_hits: set[str] = set()
                    all_have_readers = True
                    for name in set(bound_names):
                        readers = index.read_sites.get(name, {})
                        if not readers:
                            all_have_readers = False
                            break
                        for reader, sites in readers.items():
                            if reader.startswith("tests/"):
                                r_hits = _module_tests(reader, test_ids)
                            else:
                                r_hits = _tests_covering_region(coverage_map["source"], reader, [(n, n) for n in sites])
                            if d_u:
                                if reader == "sqlglot/generator.py" or reader.startswith("sqlglot/generators/"):
                                    attr_hits.update(r_hits & g_u)
                                elif reader == "sqlglot/parser.py" or reader.startswith("sqlglot/parsers/"):
                                    attr_hits.update(r_hits & p_u)
                                elif reader == "sqlglot/tokens.py":
                                    attr_hits.update(r_hits & t_u)
                                elif reader.startswith("sqlglot/dialects/") and reader != path:
                                    other_u = set(coverage_map.get("dialect_users", {}).get(reader, ()))
                                    attr_hits.update((other_u | unprof) if other_u else r_hits)
                                else:
                                    attr_hits.update(r_hits & d_u)
                            else:
                                attr_hits.update(r_hits)
                    if all_have_readers and attr_hits:
                        if d_u:
                            return attr_hits
                        selected = set(attr_hits)
                        selected.update(_module_tests(path, test_ids))
                        return selected

            selected.update(_file_tests(path, coverage_map))
            if path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
                selected.update(_component_dialect_users(path, changed_lines, coverage_map, index))
            if index is not None:
                bindings = index.scalar_bindings.get(path, {})
                for name in {bindings.get(str(line)) for line in changed_lines} - {None}:
                    for reader, sites in index.read_sites.get(name, {}).items():
                        if reader.startswith("tests/"):
                            selected.update(_module_tests(reader, test_ids))
                        else:
                            selected.update(_tests_covering_region(coverage_map["source"], reader, [(n, n) for n in sites]))

    for anchor in change["insertion_anchors"]:
        neighbors = {max(1, anchor - 1), anchor}
        affected = _function_tests(path, neighbors, coverage_map)
        if not affected:
            affected = _file_tests(path, coverage_map)
            if path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
                affected.update(_component_dialect_users(path, {anchor}, coverage_map, index))
        selected.update(affected)

    selected.update(base_extra)
    if not selected:
        selected.update(_file_tests(path, coverage_map))
    return selected or set(test_ids)


def _select_tests(
    patch: Path, coverage_map: dict, tests: list[dict], ids: list[str], index: SourceIndex | None = None
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
        analysis = index.analyze(change) if index is not None else "unknown"
        if analysis == "invalid":
            selected = set(ids)
            break
        if analysis != "equivalent":
            selected.update(_changed_source_tests(change, coverage_map, ids, index, analysis))
            if new_path and new_path != old_path and old_path:
                selected.update(_changed_source_tests({**change, "old": new_path}, coverage_map, ids, index))
    out = [test for test in tests if test["id"] in selected]
    if (
        out
        and index is not None
        and os.environ.get("SELECTOR_FAST_VERIFY") != "0"
    ):
        sq_root = Path(os.environ.get("SELECTOR_SQLGLOT_ROOT", str(APP / "sqlglot_src")))
        if len(out) == len(tests):
            extra_keep: set[str] = set()
            for ch in changes:
                p = _normalized(ch.get("old"))
                if p:
                    extra_keep.update(_file_tests(p, coverage_map))
                    if p.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
                        d_name = Path(p).stem
                        d_tests = coverage_map.get("dialect_tests", {})
                        extra_keep.update(d_tests.get(d_name, ()))
                        for sub_d in getattr(index, "dialect_subclasses", {}).get(d_name, ()):
                            extra_keep.update(d_tests.get(sub_d, ()))
            narrowed = probe_import_crash(changes, tests, index, sq_root, extra_keep_ids=extra_keep)
            if narrowed is not None:
                return narrowed
        disk_files = set(coverage_map.get("access", ()))
        _tb_m = APP / "tpcds-baseline.marshal"
        tpcds_base = _tb_m if _tb_m.is_file() else (APP / "tpcds-baseline.json")
        out = verify_commit_unified(changes, out, index, sq_root, tpcds_base, disk_read_files=disk_files)
    return out


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
        average = (total_duration + worker_count - 1) // worker_count
        if best_cost is not None:
            if longest + worker_cost >= best_cost:
                break
            if average + worker_cost >= best_cost:
                continue
        workers: list[list[int]] = [[] for _ in range(worker_count)]
        loads = [0] * worker_count
        worker_items: list[list[tuple[int, int]]] = [[] for _ in range(worker_count)]
        for number, duration in ordered:
            worker = loads.index(min(loads))
            workers[worker].append(number)
            worker_items[worker].append((number, duration))
            loads[worker] += duration
        # Fast 1-move local search to strictly improve makespan over greedy LPT when possible
        if worker_count >= 2:
            for _ in range(16):
                hi = loads.index(max(loads))
                lo = loads.index(min(loads))
                gap = loads[hi] - loads[lo]
                if gap <= 1:
                    break
                best_move = None
                best_new_max = loads[hi]
                for idx_it, (num, dur) in enumerate(worker_items[hi]):
                    if dur < gap:
                        new_lo = loads[lo] + dur
                        if new_lo < best_new_max:
                            best_new_max = new_lo
                            best_move = (idx_it, num, dur)
                if best_move is None:
                    break
                idx_it, num, dur = best_move
                worker_items[hi].pop(idx_it)
                workers[hi].pop(idx_it)
                loads[hi] -= dur
                worker_items[lo].append((num, dur))
                workers[lo].append(num)
                loads[lo] += dur
        cost = max(loads) + worker_cost
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_workers = workers
    return best_workers


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    ids = [test["id"] for test in tests]
    _cm_m = APP / "coverage-map.marshal"
    coverage_map = marshal.loads(_cm_m.read_bytes()) if _cm_m.is_file() else json.loads((APP / "coverage-map.json").read_text())
    _dp_m = APP / "dialect-profile.marshal"
    profile = marshal.loads(_dp_m.read_bytes()) if _dp_m.is_file() else (json.loads((APP / "dialect-profile.json").read_text()) if (APP / "dialect-profile.json").is_file() else json.loads(gzip.decompress((APP / "dialect-profile.json.gz").read_bytes())))
    coverage_map["dialect_users"] = profile.get("dialect_tests", {})
    coverage_map["tokenizer_users"] = profile.get("tokenizer_tests", {})
    coverage_map["parser_users"] = profile.get("parser_tests", {})
    coverage_map["generator_users"] = profile.get("generator_tests", {})
    coverage_map["dialect_attr_tests"] = profile.get("dialect_attr_tests", {})
    coverage_map["watched_dialect_attrs"] = profile.get("watched_dialect_attrs", [])
    coverage_map["dict_key_tests"] = profile.get("dict_key_tests", {})
    coverage_map["tracked_dict_attrs"] = profile.get("tracked_dict_attrs", [])
    coverage_map["dialect_unprofiled"] = profile.get("unprofiled", [])
    coverage_map["attr_unprofiled"] = profile.get("attr_unprofiled", [])
    coverage_map["lazy_import_lines"] = profile.get("lazy_import_lines", {})
    _pp_m = APP / "predicate-profile.marshal"
    pred_p = APP / "predicate-profile.json.gz"
    if _pp_m.is_file():
        coverage_map["predicate_profile"] = marshal.loads(_pp_m.read_bytes())
    elif pred_p.is_file():
        coverage_map["predicate_profile"] = json.loads(gzip.decompress(pred_p.read_bytes()))
    if "line_index" in coverage_map:
        coverage_map["source"] = {p: LineIndex(v) for p, v in coverage_map.pop("line_index").items()}
    index = SourceIndex(APP / "source-index.zip")
    for patch in sorted((DATA / "commits").glob("*.patch")):
        selected = _select_tests(patch, coverage_map, tests, ids, index)
        print(json.dumps({"commit": patch.stem, "workers": _split(selected)}))
    print("selected tests using pinned-source coverage", file=sys.stderr)


if __name__ == "__main__":
    main()
