"""Validate materialised layouts against the simulator, the truth and DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .footers import read_footer_index
from .materialize import read_manifest
from .plan import build_index, build_plan, load_columns
from .ranking import Ranking, simulate_layout
from .soundness import QueryOracle, build_oracles, check_plan
from .synth import read_tables
from .workload import Workload
from .zonemap import QueryCost, ZoneMapIndex


def compare_indexes(sim: ZoneMapIndex, real: ZoneMapIndex) -> list[str]:
    """Differences between simulated and footer statistics (files, row groups, min/max)."""
    problems = []
    if [f.partition for f in sim.files] != [f.partition for f in real.files]:
        return ["file set differs"]
    for fs, fr in zip(sim.files, real.files, strict=True):
        if len(fs.row_groups) != len(fr.row_groups):
            problems.append(
                f"{fs.partition}: {len(fs.row_groups)} vs {len(fr.row_groups)} row groups"
            )
            continue
        for i, (a, b) in enumerate(zip(fs.row_groups, fr.row_groups, strict=True)):
            if a.rows != b.rows or a.minmax != b.minmax:
                problems.append(f"{fs.partition} row group {i}: statistics differ")
    return problems


def compare_costs(sim: list[QueryCost], real: list[QueryCost]) -> list[str]:
    problems = []
    for s, r in zip(sim, real, strict=True):
        if s.files_opened != r.files_opened or sorted(s.read) != sorted(r.read):
            problems.append(
                f"{s.query_id}: simulated {s.row_groups_read} row groups, "
                f"footers {r.row_groups_read}"
            )
    return problems


def check_files(
    paths: list[Path], index: ZoneMapIndex, costs: list[QueryCost], oracles: list[QueryOracle]
) -> tuple[int, int]:
    """Re-scan each skipped row group of the real files with Polars: (violations, scanned)."""
    cache: dict[tuple[int, int], pl.DataFrame] = {}
    handles = [pq.ParquetFile(p) for p in paths]
    violations = scanned = 0
    for cost, oracle in zip(costs, oracles, strict=True):
        read = set(cost.read)
        truth = pl.Series("sale_id", oracle.truth_ids)
        for fi, f in enumerate(index.files):
            for ri in range(len(f.row_groups)):
                if (fi, ri) in read:
                    continue
                if (fi, ri) not in cache:
                    cache[(fi, ri)] = pl.from_arrow(handles[fi].read_row_group(ri))
                df = cache[(fi, ri)]
                scanned += 1
                if (
                    df.filter(oracle.model_expr).height
                    or df.filter(pl.col("sale_id").is_in(truth)).height
                ):
                    violations += 1
    return violations, scanned


def duckdb_results(
    paths: list[Path], products: pa.Table, stores: pa.Table, workload: Workload
) -> dict[str, list]:
    con = duckdb.connect()
    files = ", ".join(f"'{p}'" for p in paths)
    con.execute(
        f"CREATE VIEW sales AS SELECT * FROM read_parquet([{files}], hive_partitioning=false)"
    )
    con.register("products", products)
    con.register("stores", stores)
    out = {}
    for q in workload.queries:
        rows = con.execute(q.sql).fetchall()
        out[q.query_id] = sorted(rows, key=lambda r: tuple((v is None, str(v)) for v in r))
    con.close()
    return out


def validate_lake(data_dir: Path, lake_dir: Path, workload: Workload, ranking: Ranking) -> dict:
    sales, products, stores = read_tables(data_dir)
    baseline, layouts = read_manifest(lake_dir)
    columns = load_columns(sales)
    oracles = build_oracles(sales, products, stores, workload)
    zone_cols = workload.zone_columns
    bpr = ranking.bytes_per_row
    reference: dict[str, list] | None = None
    reports = []
    # the baseline lake is the reference for query results, so it goes first
    ordered = [baseline] + [lay for lay in layouts if lay != baseline]
    for layout in ordered:
        root = lake_dir / layout.layout_id
        plan = build_plan(columns, sales.num_rows, layout)
        sim_index = build_index(plan, columns, zone_cols)
        sim_costs = simulate_layout(sim_index, workload, bpr)
        real_index, paths = read_footer_index(root, layout, zone_cols)
        real_costs = simulate_layout(real_index, workload, bpr)
        mismatches = compare_indexes(sim_index, real_index) + compare_costs(sim_costs, real_costs)
        model_violations = check_plan(plan, sim_costs, oracles)
        file_violations, scanned = check_files(paths, real_index, real_costs, oracles)
        results = duckdb_results(paths, products, stores, workload)
        if reference is None:
            reference = results
        result_mismatches = [qid for qid in results if results[qid] != reference[qid]]
        reports.append(
            {
                **layout.to_dict(),
                "is_baseline": layout == baseline,
                "files": len(paths),
                "row_groups": real_index.n_row_groups,
                "row_groups_rescanned": scanned,
                "soundness_violations": model_violations + file_violations,
                "simulator_footer_mismatches": len(mismatches),
                "mismatch_detail": mismatches[:10],
                "duckdb_result_mismatches": len(result_mismatches),
                "duckdb_mismatched_queries": result_mismatches,
            }
        )
    return {
        "queries": len(workload.queries),
        "extractor_violations": sum(o.extractor_violations for o in oracles),
        "layouts": reports,
        "summary": {
            "soundness_violations": sum(r["soundness_violations"] for r in reports),
            "simulator_footer_mismatches": sum(r["simulator_footer_mismatches"] for r in reports),
            "duckdb_result_mismatches": sum(r["duckdb_result_mismatches"] for r in reports),
            "extractor_violations": sum(o.extractor_violations for o in oracles),
            "row_groups_rescanned": sum(r["row_groups_rescanned"] for r in reports),
        },
    }


def is_clean(report: dict) -> bool:
    s = report["summary"]
    return not (
        s["soundness_violations"]
        or s["simulator_footer_mismatches"]
        or s["duckdb_result_mismatches"]
        or s["extractor_violations"]
    )
