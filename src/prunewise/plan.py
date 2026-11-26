"""Turn a layout into files and row groups, and compute per-row-group min/max statistics.

``split_row_groups`` is the single function that decides where row groups start and end. The
simulator and the materialiser both use it, so the model and the written files cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from .layouts import Layout
from .schema import kind_of_arrow
from .zonemap import FileStats, RowGroupStats, ZoneMapIndex


def load_columns(table: pa.Table, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Canonical numpy arrays: ints and dates (epoch days) as int64, floats, strings as objects."""
    out: dict[str, np.ndarray] = {}
    for name in names or table.column_names:
        col = table.column(name).combine_chunks()
        kind = kind_of_arrow(col.type)
        if kind == "date":
            out[name] = col.cast(pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64)
        elif kind == "int":
            out[name] = col.to_numpy(zero_copy_only=False).astype(np.int64)
        elif kind == "float":
            out[name] = col.to_numpy(zero_copy_only=False).astype(np.float64)
        elif kind == "str":
            out[name] = np.array(col.to_pylist(), dtype=object)
    return out


def split_row_groups(n_rows: int, row_group_size: int) -> list[tuple[int, int]]:
    return [(s, min(s + row_group_size, n_rows)) for s in range(0, n_rows, row_group_size)]


@dataclass
class FilePlan:
    partition: tuple  # canonical partition values, in the layout's partition-column order
    rows: np.ndarray  # row indices of the source table, in file order


@dataclass
class LayoutPlan:
    layout: Layout
    files: list[FilePlan]

    @property
    def n_rows(self) -> int:
        return sum(len(f.rows) for f in self.files)

    @property
    def n_row_groups(self) -> int:
        return sum(
            len(split_row_groups(len(f.rows), self.layout.row_group_size)) for f in self.files
        )


def build_plan(columns: dict[str, np.ndarray], n_rows: int, layout: Layout) -> LayoutPlan:
    if layout.partition:
        uniques, codes = [], []
        for name in layout.partition:
            u, inv = np.unique(columns[name], return_inverse=True)
            uniques.append(u)
            codes.append(inv.astype(np.int64))
        key = np.zeros(n_rows, dtype=np.int64)
        for u, c in zip(uniques, codes, strict=True):
            key = key * len(u) + c
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        cuts = np.flatnonzero(np.diff(sorted_key)) + 1
        groups = np.split(order, cuts)
    else:
        groups = [np.arange(n_rows)]
    files = []
    for rows in groups:
        first = rows[0]
        values = tuple(_scalar(columns[name][first]) for name in layout.partition)
        if layout.sort is not None:
            rows = rows[np.argsort(columns[layout.sort][rows], kind="stable")]
        files.append(FilePlan(values, rows))
    files.sort(key=lambda f: f.partition)
    return LayoutPlan(layout, files)


def _scalar(v):
    return v.item() if hasattr(v, "item") else v


def build_index(
    plan: LayoutPlan, columns: dict[str, np.ndarray], zone_cols: list[str]
) -> ZoneMapIndex:
    """Per-row-group min/max for ``zone_cols`` without writing a file."""
    names = plan.layout.partition
    files = []
    for f in plan.files:
        groups = []
        for start, stop in split_row_groups(len(f.rows), plan.layout.row_group_size):
            idx = f.rows[start:stop]
            minmax = {}
            for col in zone_cols:
                vals = columns[col][idx]
                minmax[col] = (_scalar(vals.min()), _scalar(vals.max()))
            groups.append(RowGroupStats(stop - start, minmax))
        files.append(FileStats(dict(zip(names, f.partition, strict=True)), groups))
    return ZoneMapIndex(names, files)
