"""Soundness: a row group the model skips must not hold a row that the query matches.

Two independent references decide which rows match a query:
  * the truth: the real SQL, run in DuckDB and rewritten to return the ``sale_id`` of every row;
  * the model: the extracted per-column predicate, evaluated as a Polars expression.
The truth must be a subset of the model (the extractor only ever loosens a query), and no skipped
row group may contain a row of either.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa

from .intervals import IntervalSet
from .plan import LayoutPlan, split_row_groups
from .predicates import truth_sql
from .schema import FACT_COLUMNS, days_to_date
from .workload import Workload
from .zonemap import QueryCost


def polars_predicate(restrictions: dict[str, IntervalSet]) -> pl.Expr:
    """The model's predicate as a Polars expression over the original (typed) columns."""
    per_col = []
    for col, rs in restrictions.items():
        kind = FACT_COLUMNS[col].kind

        def lit(v, kind=kind):
            return days_to_date(v) if kind == "date" else v

        alts = []
        for lo, lo_inc, hi, hi_inc in rs.intervals:
            conds = []
            if lo is not None:
                conds.append(pl.col(col) >= lit(lo) if lo_inc else pl.col(col) > lit(lo))
            if hi is not None:
                conds.append(pl.col(col) <= lit(hi) if hi_inc else pl.col(col) < lit(hi))
            alts.append(pl.all_horizontal(conds) if conds else pl.lit(True))
        per_col.append(pl.any_horizontal(alts) if alts else pl.lit(False))
    return pl.all_horizontal(per_col) if per_col else pl.lit(True)


@dataclass
class QueryOracle:
    query_id: str
    truth_ids: np.ndarray  # sale_id of every row the real query reads
    model_expr: pl.Expr
    truth_mask: np.ndarray  # over source-table rows
    model_mask: np.ndarray

    @property
    def extractor_violations(self) -> int:
        """Rows the real query matches that the model would have excluded."""
        return int((self.truth_mask & ~self.model_mask).sum())

    @property
    def any_mask(self) -> np.ndarray:
        return self.truth_mask | self.model_mask


def build_oracles(
    sales: pa.Table, products: pa.Table, stores: pa.Table, workload: Workload
) -> list[QueryOracle]:
    con = duckdb.connect()
    con.register("sales", sales)
    con.register("products", products)
    con.register("stores", stores)
    frame = pl.from_arrow(sales)
    sale_ids = sales.column("sale_id").to_numpy()
    out = []
    for q in workload.queries:
        ids = np.array([r[0] for r in con.execute(truth_sql(q.sql)).fetchall()], dtype=np.int64)
        expr = polars_predicate(q.restrictions)
        model_mask = frame.with_columns(expr.alias("m")).get_column("m").to_numpy()
        out.append(QueryOracle(q.query_id, ids, expr, np.isin(sale_ids, ids), model_mask))
    con.close()
    return out


def skipped_with_match(plan: LayoutPlan, cost: QueryCost, mask: np.ndarray) -> int:
    """Row groups the simulator skipped that hold at least one row of ``mask``."""
    read = set(cost.read)
    bad = 0
    for fi, f in enumerate(plan.files):
        flags = mask[f.rows]
        for ri, (start, stop) in enumerate(
            split_row_groups(len(f.rows), plan.layout.row_group_size)
        ):
            if (fi, ri) not in read and flags[start:stop].any():
                bad += 1
    return bad


def check_plan(plan: LayoutPlan, costs: list[QueryCost], oracles: list[QueryOracle]) -> int:
    """Total skipped-but-matching row groups for one layout, across all queries."""
    return sum(skipped_with_match(plan, c, o.any_mask) for c, o in zip(costs, oracles, strict=True))
