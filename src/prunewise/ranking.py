"""Enumerate layouts, simulate every query on each and rank by weighted cost."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

from .bytesmodel import measure_bytes_per_row
from .config import Config
from .layouts import (
    Layout,
    baseline_layout,
    enumerate_candidates,
    exclusion_reason,
    expected_file_count,
    small_file_flag,
)
from .plan import LayoutPlan, build_index, build_plan, load_columns
from .workload import Workload
from .zonemap import QueryCost, ZoneMapIndex, simulate_query, weighted_cost


@dataclass
class LayoutEvaluation:
    layout: Layout
    files: int
    row_groups: int
    excluded: bool = False
    exclusion_reason: str | None = None
    flags: list[str] = field(default_factory=list)
    costs: list[QueryCost] = field(default_factory=list)
    cost: float = 0.0
    bytes_per_day: float = 0.0
    ratio_to_baseline: float | None = None
    is_baseline: bool = False

    def to_dict(self) -> dict:
        return {
            **self.layout.to_dict(),
            "is_baseline": self.is_baseline,
            "files": self.files,
            "row_groups": self.row_groups,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
            "flags": self.flags,
            "cost": self.cost,
            "bytes_per_day": self.bytes_per_day,
            "ratio_to_baseline": self.ratio_to_baseline,
            "queries": [c.to_dict() for c in self.costs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> LayoutEvaluation:
        return cls(
            layout=Layout.from_dict(d),
            files=d["files"],
            row_groups=d["row_groups"],
            excluded=d["excluded"],
            exclusion_reason=d["exclusion_reason"],
            flags=d["flags"],
            costs=[QueryCost(**q, read=[]) for q in d["queries"]],
            cost=d["cost"],
            bytes_per_day=d["bytes_per_day"],
            ratio_to_baseline=d["ratio_to_baseline"],
            is_baseline=d["is_baseline"],
        )


@dataclass
class Ranking:
    config: Config
    n_rows: int
    bytes_per_row: dict[str, float]
    baseline: LayoutEvaluation
    candidates: list[LayoutEvaluation]  # ranked: cheapest first, excluded layouts last

    @property
    def ranked(self) -> list[LayoutEvaluation]:
        return [c for c in self.candidates if not c.excluded]

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "n_rows": self.n_rows,
            "bytes_per_row": self.bytes_per_row,
            "baseline": self.baseline.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Ranking:
        return cls(
            Config.from_dict(d["config"]),
            d["n_rows"],
            d["bytes_per_row"],
            LayoutEvaluation.from_dict(d["baseline"]),
            [LayoutEvaluation.from_dict(c) for c in d["candidates"]],
        )


def simulate_layout(
    index: ZoneMapIndex, workload: Workload, bytes_per_row: dict[str, float]
) -> list[QueryCost]:
    return [
        simulate_query(
            index,
            q.restrictions,
            q.referenced_columns,
            bytes_per_row,
            q.query_id,
            q.executions_per_day,
        )
        for q in workload.queries
    ]


def evaluate_layout(
    layout: Layout,
    columns: dict[str, np.ndarray],
    n_rows: int,
    cardinality: dict[str, int],
    workload: Workload,
    bytes_per_row: dict[str, float],
    cfg: Config,
    check_exclusion: bool = True,
) -> tuple[LayoutEvaluation, LayoutPlan | None, ZoneMapIndex | None]:
    expected = expected_file_count(layout, cardinality, n_rows)
    reason = exclusion_reason(layout, expected, n_rows, cfg) if check_exclusion else None
    if reason:
        ev = LayoutEvaluation(
            layout, expected, 0, excluded=True, exclusion_reason=reason, flags=[reason]
        )
        return ev, None, None
    plan = build_plan(columns, n_rows, layout)
    index = build_index(plan, columns, workload.zone_columns)
    costs = simulate_layout(index, workload, bytes_per_row)
    flags = []
    sizes = [len(f.rows) for f in plan.files]
    flag = small_file_flag(float(np.median(sizes)), cfg)
    if flag:
        flags.append(flag)
    ev = LayoutEvaluation(
        layout,
        len(plan.files),
        plan.n_row_groups,
        flags=flags,
        costs=costs,
        cost=weighted_cost(costs, cfg, len(plan.files), bool(flag)),
        bytes_per_day=sum(c.frequency * c.bytes_read for c in costs),
    )
    return ev, plan, index


def cardinalities(columns: dict[str, np.ndarray], names: list[str]) -> dict[str, int]:
    return {n: len(np.unique(columns[n])) for n in names}


def rank_layouts(
    table: pa.Table,
    workload: Workload,
    cfg: Config,
    bytes_per_row: dict[str, float] | None = None,
) -> Ranking:
    columns = load_columns(table)
    n_rows = table.num_rows
    bpr = bytes_per_row or measure_bytes_per_row(table, cfg.baseline_row_group_size)
    card = cardinalities(columns, ["sale_date", "sale_month", "store_region", "store_id"])
    baseline, _, _ = evaluate_layout(
        baseline_layout(cfg), columns, n_rows, card, workload, bpr, cfg, check_exclusion=False
    )
    baseline.is_baseline = True
    evaluated = [
        evaluate_layout(lay, columns, n_rows, card, workload, bpr, cfg)[0]
        for lay in enumerate_candidates(cfg)
    ]
    for ev in evaluated:
        if not ev.excluded and baseline.cost > 0:
            ev.ratio_to_baseline = ev.cost / baseline.cost
    baseline.ratio_to_baseline = 1.0 if baseline.cost > 0 else None
    evaluated.sort(key=lambda e: (e.excluded, e.cost, e.layout.layout_id))
    return Ranking(cfg, n_rows, bpr, baseline, evaluated)
