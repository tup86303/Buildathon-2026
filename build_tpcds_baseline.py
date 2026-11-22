"""Pre-compute deterministic (optimized_sql_hash, duckdb_sql_hash) for the 57 executed TPC-DS queries."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    sqlglot_root = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()
    os.chdir(sqlglot_root)
    sys.path.insert(0, str(sqlglot_root))

    from sqlglot import parse_one, transpile
    from sqlglot.optimizer import optimize
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

    entries: list[dict] = []
    pairs = [(i, m, s) for i, (m, s, _) in enumerate(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql")) if m.get("execute")]
    pairs.sort(key=lambda x: len(x[2]))
    for idx, meta, sql in pairs:
        opt_sql = optimize(parse_one(sql), TPCDS_SCHEMA, leave_tables_isolated=True).sql()
        duck_sql = transpile(sql, write="duckdb")[0]
        entries.append({
            "idx": idx,
            "opt_hash": hashlib.blake2b(opt_sql.encode("utf-8"), digest_size=16).hexdigest(),
            "duck_hash": hashlib.blake2b(duck_sql.encode("utf-8"), digest_size=16).hexdigest(),
        })

    tpch_h = hashlib.blake2b(digest_size=16)
    for _, sql, opt in load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"):
        tpch_h.update(transpile(sql, write="duckdb")[0].encode("utf-8"))
        tpch_h.update(b"\0")
        tpch_h.update(transpile(opt, write="duckdb")[0].encode("utf-8"))
        tpch_h.update(b"\n")

    out_path.write_text(
        json.dumps({"queries": entries, "tpch_opt_hash": tpch_h.hexdigest()}, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"wrote {len(entries)} TPC-DS + TPC-H opt baseline hashes to {out_path}")


if __name__ == "__main__":
    main()
