"""Candidate layouts: partition key, sort key within files and row-group size."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .config import Config

PARTITION_OPTIONS: list[tuple[str, ...]] = [
    (),
    ("sale_month",),
    ("store_region",),
    ("store_region", "sale_month"),
    ("store_id", "sale_date"),
]
SORT_OPTIONS: list[str | None] = [None, "sale_date", "product_id", "store_id"]
BASELINE_PARTITION = ("sale_date",)


@dataclass(frozen=True)
class Layout:
    partition: tuple[str, ...]
    sort: str | None
    row_group_size: int

    @property
    def layout_id(self) -> str:
        part = "-".join(self.partition) or "none"
        return f"p_{part}__s_{self.sort or 'none'}__rg{self.row_group_size}"

    def to_dict(self) -> dict:
        return {
            "layout_id": self.layout_id,
            "partition": list(self.partition),
            "sort": self.sort,
            "row_group_size": self.row_group_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Layout:
        return cls(tuple(d["partition"]), d["sort"], d["row_group_size"])


def baseline_layout(cfg: Config) -> Layout:
    """What the lake does today: one file per day, arrival order, one large row-group size."""
    return Layout(BASELINE_PARTITION, None, cfg.baseline_row_group_size)


def enumerate_candidates(cfg: Config) -> list[Layout]:
    return [
        Layout(p, s, rg)
        for p in PARTITION_OPTIONS
        for s in SORT_OPTIONS
        for rg in cfg.row_group_sizes
    ]


def expected_file_count(layout: Layout, cardinality: dict[str, int], n_rows: int) -> int:
    """Upper bound on files written: one per distinct partition value, at most one per row."""
    if not layout.partition:
        return 1
    return min(prod(cardinality[c] for c in layout.partition), n_rows)


def exclusion_reason(layout: Layout, expected_files: int, n_rows: int, cfg: Config) -> str | None:
    if expected_files > cfg.max_files:
        return (
            f"small-file rule: expected {expected_files} files (median about "
            f"{n_rows // expected_files} rows) exceeds the limit of {cfg.max_files} files"
        )
    return None


def small_file_flag(median_rows: float, cfg: Config) -> str | None:
    if median_rows < cfg.min_file_rows:
        return (
            f"small-file rule: median file has {int(median_rows)} rows, below the "
            f"{cfg.min_file_rows}-row minimum; listing penalty applied"
        )
    return None
