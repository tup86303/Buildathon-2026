"""Generic pure-predicate AST diff matcher and restricted evaluator for pruning expensive
tests whose observed runtime operand states produce identical boolean decisions on old vs new."""
from __future__ import annotations

import ast
from typing import Any

# Base hierarchy of sqlglot.exp classes built lazily (pure static MRO check, no side effects)
_SUBCLASS_MAP: dict[str, set[str]] | None = None

_ALLOWED_PROPS = frozenset({
    "quoted", "is_string", "is_int", "is_number", "is_star",
    "side", "method", "type", "name",
})


def _extract_single_predicate_edit(before_tree: ast.Module, after_tree: ast.Module) -> tuple[int, ast.AST, ast.AST] | None:
    """If before_tree and after_tree differ ONLY in a single boolean expression inside an
    ast.If.test or ast.Return.value (same structure everywhere else), return (lineno, old_expr, new_expr)."""
    # Collect all (If.test / Return.value) nodes in parallel traversal
    b_nodes = list(ast.walk(before_tree))
    a_nodes = list(ast.walk(after_tree))
    # Fast structural check: find candidate parent statements at same position
    b_stmts = [n for n in b_nodes if isinstance(n, (ast.If, ast.Return)) and (n.test if isinstance(n, ast.If) else n.value) is not None]
    a_stmts = [n for n in a_nodes if isinstance(n, (ast.If, ast.Return)) and (n.test if isinstance(n, ast.If) else n.value) is not None]
    if len(b_stmts) != len(a_stmts):
        return None
    diff_pair = None
    for bs, as_ in zip(b_stmts, a_stmts):
        if type(bs) is not type(as_):
            return None
        be = bs.test if isinstance(bs, ast.If) else bs.value
        ae = as_.test if isinstance(as_, ast.If) else as_.value
        if ast.dump(be, include_attributes=False) != ast.dump(ae, include_attributes=False):
            if diff_pair is not None:
                return None  # more than one predicate edited
            diff_pair = (bs, as_, be, ae)
    if diff_pair is None:
        return None
    bs, as_, be, ae = diff_pair
    # Verify NOTHING outside (be, ae) changed by temporarily swapping in ast.Constant(True)
    sentinel = ast.Constant(value=True)
    if isinstance(bs, ast.If):
        bs.test = sentinel; as_.test = sentinel
    else:
        bs.value = sentinel; as_.value = sentinel
    try:
        if hasattr(before_tree, "type_ignores"): before_tree.type_ignores = []
        if hasattr(after_tree, "type_ignores"): after_tree.type_ignores = []
        same = ast.dump(before_tree, include_attributes=False) == ast.dump(after_tree, include_attributes=False)
    finally:
        if isinstance(bs, ast.If):
            bs.test = be; as_.test = ae
        else:
            bs.value = be; as_.value = ae
    if not same:
        return None
    return (bs.lineno, be, ae)


def _extract_type_names(node: ast.AST) -> set[str] | None:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Tuple):
        out: set[str] = set()
        for elt in node.elts:
            sub = _extract_type_names(elt)
            if sub is None:
                return None
            out |= sub
        return out
    return None


def _eval_pure(node: ast.AST, state: dict[str, Any], mro_map: dict[str, set[str]]) -> tuple[bool, Any]:
    """Evaluate a strictly pure boolean/attribute expression against a recorded state dict.
    Returns (ok, value). If any construct or attribute is unmodeled, returns (False, None)."""
    if isinstance(node, ast.Constant):
        return (True, node.value)
    if isinstance(node, ast.Name):
        if node.id in state:
            return (True, state[node.id])
        return (False, None)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        ok, v = _eval_pure(node.operand, state, mro_map)
        return (ok, not v) if ok else (False, None)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            val: Any = True
            for vnode in node.values:
                ok, val = _eval_pure(vnode, state, mro_map)
                if not ok:
                    return (False, None)
                if not val:
                    return (True, val)
            return (True, val)
        if isinstance(node.op, ast.Or):
            val = False
            for vnode in node.values:
                ok, val = _eval_pure(vnode, state, mro_map)
                if not ok:
                    return (False, None)
                if val:
                    return (True, val)
            return (True, val)
        return (False, None)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        ok1, lv = _eval_pure(node.left, state, mro_map)
        ok2, rv = _eval_pure(node.comparators[0], state, mro_map)
        if not (ok1 and ok2):
            return (False, None)
        op = node.ops[0]
        if isinstance(op, ast.Eq): return (True, lv == rv)
        if isinstance(op, ast.NotEq): return (True, lv != rv)
        if isinstance(op, ast.Is): return (True, lv is rv)
        if isinstance(op, ast.IsNot): return (True, lv is not rv)
        return (False, None)
    if isinstance(node, ast.Call):
        # Support isinstance(x, Cls) and x.args.get("key")
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance" and len(node.args) == 2 and not node.keywords:
            if not isinstance(node.args[0], ast.Name):
                return (False, None)
            var = node.args[0].id
            tkey = f"{var}.__type__"
            if tkey not in state:
                return (False, None)
            tnames = _extract_type_names(node.args[1])
            if not tnames:
                return (False, None)
            actual_t = state[tkey]
            mro = mro_map.get(actual_t, {actual_t})
            return (True, bool(mro & tnames))
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "args"
            and isinstance(node.func.value.value, ast.Name)
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and not node.keywords
        ):
            k = f"{node.func.value.value.id}.args[{node.args[0].value}]"
            if k in state:
                return (True, state[k])
            return (False, None)
        return (False, None)
    if isinstance(node, ast.Subscript):
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "args"
            and isinstance(node.value.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            k = f"{node.value.value.id}.args[{node.slice.value}]"
            if k in state:
                return (True, state[k])
        return (False, None)
    if isinstance(node, ast.Attribute):
        # Support var.prop and var.this.prop
        if isinstance(node.value, ast.Name):
            k = f"{node.value.id}.{node.attr}"
            if k in state:
                return (True, state[k])
        elif isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
            k = f"{node.value.value.id}.{node.value.attr}.{node.attr}"
            if k in state:
                return (True, state[k])
        return (False, None)
    return (False, None)


def prune_unaffected_tests(
    path: str,
    before_tree: ast.Module,
    after_tree: ast.Module,
    selected: set[str],
    pred_profile: dict,
) -> set[str]:
    """Prune tests from `selected` ONLY when:
    1. The file diff modifies ONLY a single pure boolean predicate in an If.test or Return.value.
    2. For a candidate test, `pred_profile` has a COMPLETE (non-'*') state set at (path, lineno).
    3. Both old and new predicates evaluate purely and agree (bool(old) == bool(new)) on 100% of states."""
    if not pred_profile or not selected:
        return selected
    sites = pred_profile.get("sites", {}).get(path)
    if not sites:
        return selected
    edit = _extract_single_predicate_edit(before_tree, after_tree)
    if edit is None:
        return selected
    lineno, old_expr, new_expr = edit
    site_tests = sites.get(str(lineno))
    if not site_tests:
        return selected
    mro_map = {k: set(v) for k, v in pred_profile.get("mro", {}).items()}
    unprofiled = set(pred_profile.get("unprofiled", ()))
    pruned = set(selected)
    for tid in list(selected):
        if tid in unprofiled:
            continue
        states = site_tests.get(tid)
        if states is None or states == "*":
            continue
        # Empty state list (never reached this exact predicate line despite covering file) -> unaffected!
        all_agree = True
        for st in states:
            ok1, v1 = _eval_pure(old_expr, st, mro_map)
            ok2, v2 = _eval_pure(new_expr, st, mro_map)
            if not (ok1 and ok2) or bool(v1) != bool(v2):
                all_agree = False
                break
        if all_agree:
            pruned.discard(tid)
    return pruned
